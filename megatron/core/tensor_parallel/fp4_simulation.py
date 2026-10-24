# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Eager fake-quantization helpers for NVFP4 and Averis experiments.

This module emulates the numerical effect of NVFP4 on Megatron's local linear
layers. It does not use packed FP4 storage or native FP4 GEMM kernels.
"""

import os

import torch
import torch.nn.functional as F


_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_NVFP4_BLOCK_SIZE = 16
_E4M3_MAX = 448.0
_E4M3_MIN_SUBNORMAL = 2.0**-9


def get_fp4_simulation_mode() -> str | None:
    """Return the requested fake-quantization mode from the environment."""
    mode = os.environ.get("MEGATRON_FP4_SIM", "").strip().lower()
    if not mode or mode in ("0", "off", "none", "false"):
        return None
    if mode not in ("nvfp4", "averis"):
        raise ValueError(
            "MEGATRON_FP4_SIM must be unset, 'nvfp4', or 'averis'; " f"got {mode!r}."
        )
    return mode


def _e2m1_quantize(values: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """Round values to the signed finite values represented by FP4 E2M1."""
    levels = torch.tensor(_FP4_LEVELS, dtype=values.dtype, device=values.device)
    magnitudes = values.abs().clamp(max=_FP4_LEVELS[-1])

    if stochastic:
        upper = torch.bucketize(magnitudes.contiguous(), levels[1:])
        lower_values = levels[upper]
        upper_values = levels[(upper + 1).clamp(max=len(_FP4_LEVELS) - 1)]
        denominator = (upper_values - lower_values).clamp_min(torch.finfo(values.dtype).eps)
        upper_probability = ((magnitudes - lower_values) / denominator).clamp(0.0, 1.0)
        rounded = torch.where(
            torch.rand_like(magnitudes) < upper_probability, upper_values, lower_values
        )
    else:
        midpoints = torch.tensor(_FP4_MIDPOINTS, dtype=values.dtype, device=values.device)
        indices = torch.bucketize(magnitudes.contiguous(), midpoints)
        rounded = levels[indices]

    return torch.sign(values) * rounded


def nvfp4_fake_quantize(
    tensor: torch.Tensor, *, stochastic: bool = False, block_size: int = _NVFP4_BLOCK_SIZE
) -> torch.Tensor:
    """Quantize and dequantize using NVFP4-style E2M1 + E4M3 block scales.

    The last dimension is divided into blocks of 16 values. Each block has an
    E4M3 scale, and a tensor-wide scale keeps the largest block scale in range.
    The returned tensor has the original shape and dtype.
    """
    if tensor.numel() == 0:
        return tensor
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    matrix = tensor.reshape(-1, original_shape[-1]).float()
    padded_size = ((matrix.size(-1) + block_size - 1) // block_size) * block_size
    if padded_size != matrix.size(-1):
        matrix = F.pad(matrix, (0, padded_size - matrix.size(-1)))

    blocks = matrix.reshape(matrix.size(0), -1, block_size)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    min_tensor_amax = (_E4M3_MAX * _FP4_LEVELS[-1]) / (torch.finfo(torch.float32).max / 4)
    tensor_amax = block_amax.amax().clamp_min(min_tensor_amax)

    # NVFP4 uses an outer scale and per-block E4M3 scale. Scaling the largest
    # local scale to E4M3's finite maximum avoids overflow in the local scales.
    global_scale = (_E4M3_MAX * _FP4_LEVELS[-1]) / tensor_amax
    local_scale = (block_amax * global_scale / _FP4_LEVELS[-1]).clamp(
        min=_E4M3_MIN_SUBNORMAL, max=_E4M3_MAX
    )
    local_scale = local_scale.to(torch.float8_e4m3fn).float()
    local_scale = local_scale.clamp_min(_E4M3_MIN_SUBNORMAL)

    scaled = blocks * global_scale / local_scale
    quantized = _e2m1_quantize(scaled, stochastic=stochastic)
    dequantized = quantized * local_scale / global_scale
    dequantized = dequantized.reshape(matrix.size(0), padded_size)[:, : original_shape[-1]]
    return dequantized.reshape(original_shape).to(original_dtype)


def _matrix_mean_residual(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a token-row mean and residual, accumulating the reduction in FP32."""
    mean = matrix.float().mean(dim=0).to(matrix.dtype)
    return mean, matrix - mean


def fp4_linear_forward(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mode: str,
) -> torch.Tensor:
    """Fake-quantized linear forward, where weight is stored as [out_features, in_features]."""
    input_matrix = input_tensor.reshape(-1, input_tensor.size(-1))
    if mode == "nvfp4":
        output = nvfp4_fake_quantize(input_matrix) @ nvfp4_fake_quantize(weight).t()
    else:
        mean, residual = _matrix_mean_residual(input_matrix)
        quantized_mean = nvfp4_fake_quantize(mean.unsqueeze(0))
        quantized_residual = nvfp4_fake_quantize(residual)
        quantized_weight = nvfp4_fake_quantize(weight)
        output = quantized_residual @ quantized_weight.t()
        output = output + (quantized_mean @ quantized_weight.t()).expand_as(output)

    output = output.reshape(*input_tensor.shape[:-1], weight.size(0))
    if bias is not None:
        output = output + bias
    return output


def fp4_linear_backward(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    mode: str,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake-quantized input- and weight-gradient GEMMs for a linear layer."""
    input_matrix = input_tensor.reshape(-1, input_tensor.size(-1))
    grad_matrix = grad_output.reshape(-1, grad_output.size(-1))
    quantized_weight = nvfp4_fake_quantize(weight)

    if mode == "nvfp4":
        quantized_grad = nvfp4_fake_quantize(grad_matrix, stochastic=stochastic_rounding)
        grad_input = quantized_grad @ quantized_weight
        grad_weight = quantized_grad.t() @ nvfp4_fake_quantize(input_matrix)
    else:
        input_mean, input_residual = _matrix_mean_residual(input_matrix)
        grad_mean, grad_residual = _matrix_mean_residual(grad_matrix)
        quantized_input_mean = nvfp4_fake_quantize(input_mean.unsqueeze(0))
        quantized_input_residual = nvfp4_fake_quantize(input_residual)
        quantized_grad_mean = nvfp4_fake_quantize(
            grad_mean.unsqueeze(0), stochastic=stochastic_rounding
        )
        quantized_grad_residual = nvfp4_fake_quantize(
            grad_residual, stochastic=stochastic_rounding
        )

        grad_input = (quantized_grad_mean @ quantized_weight).expand_as(input_matrix)
        grad_input = grad_input + quantized_grad_residual @ quantized_weight
        token_count = input_matrix.size(0)
        grad_weight = quantized_grad_residual.t() @ quantized_input_residual
        grad_weight = grad_weight + token_count * (
            quantized_grad_mean.t() @ quantized_input_mean
        )

    return grad_input.reshape_as(input_tensor), grad_weight
