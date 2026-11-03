"""`--log-probs-backend fused` on `PaddedRows`: the padded `[T, V]` logits scored by one compiled program per shape.

XLA fuses the fp32 upcast into the reductions, so no fp32 `[T, V]` copy, softmax or log-softmax is materialized or
saved: the backward keeps only the bf16 logits and the `[T]` logsumexp. On CUDA the same flag selects the Triton
kernel instead (`logit_processors.py`).
"""

import torch

from miles.platforms import current_platform


def _compiled(fn):
    # one program per padded shape where shapes are static; eager elsewhere (the CPU tests)
    if current_platform.dynamic_shapes:
        return fn
    return torch.compile(fn, backend=current_platform.compile_backend, fullgraph=True, dynamic=False)


@_compiled
def _forward(logits: torch.Tensor, targets: torch.Tensor, inv_temperature: torch.Tensor):
    x = logits.float() * inv_temperature
    lse = torch.logsumexp(x, dim=-1)
    log_probs = x.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - lse
    entropy = lse - (torch.exp(x - lse.unsqueeze(-1)) * x).sum(-1)
    return log_probs, entropy, lse


@_compiled
def _backward(
    grad: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, lse: torch.Tensor, inv_temperature: torch.Tensor
):
    x = logits.float() * inv_temperature
    one_hot = torch.nn.functional.one_hot(targets, logits.size(-1)).to(torch.float32)
    return ((one_hot - torch.exp(x - lse.unsqueeze(-1))) * (grad * inv_temperature).unsqueeze(-1)).to(logits.dtype)


class FusedRowLogProb(torch.autograd.Function):
    """Log-probs of `targets` `[T]` and entropy `[T]` for `[T, V]` logits; entropy carries no gradient."""

    @staticmethod
    def forward(ctx, logits, targets, temperature):
        inv_t = torch.full(
            (), 1.0 / temperature if temperature > 0 else 1.0, dtype=torch.float32, device=logits.device
        )
        log_probs, entropy, lse = _forward(logits, targets, inv_t)
        ctx.save_for_backward(logits, targets, lse, inv_t)
        ctx.mark_non_differentiable(entropy)
        return log_probs, entropy

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, targets, lse, inv_t = ctx.saved_tensors
        return _backward(grad_log_probs.contiguous(), logits, targets, lse, inv_t), None, None
