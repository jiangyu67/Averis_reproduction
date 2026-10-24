# Qwen3-0.6B FP4 simulation pilot

This pilot applies eager fake quantization to Megatron's local tensor-parallel
linear layers. It is intended to compare short-run training quality; it does
not store FP4 weights or run native FP4 kernels.

## NVFP4-style simulation parameters

- FP4 values use signed E2M1 levels `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
- Values are blocked in groups of 16 along the last matrix dimension.
- Each block's scale is rounded to E4M3; an outer tensor scale keeps the
  largest block scale within E4M3's finite range.
- Forward operands use deterministic nearest-level rounding.
- Backward output gradients use stochastic rounding by default, following the
  paper's FP4 training setup. Set `MEGATRON_FP4_STOCHASTIC_ROUNDING=0` to turn
  it off for a deterministic diagnostic run.
- Parameters, optimizer states, and GEMM accumulations remain BF16/FP32 as in
  the normal Megatron path. Fake quantization therefore does not reproduce
  FP4 memory savings or native FP4 throughput.

This is an NVFP4-style software approximation, not a bit-exact reproduction of
Transformer Engine's scale selection, packing, or hardware rounding behavior.

## Averis mode

Set `MEGATRON_FP4_SIM=averis`. For each local linear layer, the implementation
splits the input rows into a feature-wise mean and residual, quantizes those
separately, and computes their forward GEMMs with the quantized weight. In the
backward pass it splits the output gradient in the same way; the input-gradient
GEMM uses both quantized pieces, and the weight-gradient GEMM uses
`Q(D_R)^T Q(X_R) + L Q(mu_D)^T Q(mu_X)`.

## Scope and restrictions

- The dispatch covers trainable linear layers that use Megatron's local
  `linear_with_grad_accumulation_and_async_allreduce` path. This includes the
  Qwen attention/MLP projections and output projection.
- Standalone attention score/value products, softmax, normalization, and
  non-linear operations are not fake-quantized. This is therefore not yet a
  full W4A4G4 simulation of every operation in the model.
- Use `--transformer-impl local`, disable gradient accumulation fusion, and
  leave `--fp16-lm-cross-entropy` disabled. Deferred embedding weight gradients
  and GTP weight rematerialization are rejected explicitly.
- The 1k-step WikiText-103 run is a short diagnostic. The paper's headline
  quality results use much larger DCLM token budgets and separate downstream
  evaluation, so this run cannot establish paper-level alignment by itself.

## Launch

From the repository root in the training container:

```bash
MEGATRON_FP4_SIM=nvfp4 bash examples/averis/qwen3_0_6b_wikitext103_1k.sh
MEGATRON_FP4_SIM=averis bash examples/averis/qwen3_0_6b_wikitext103_1k.sh
```

The script writes TensorBoard events under `/workspace/log/<run-name>` and
checkpoints under `/mnt/llm/checkpoints/<run-name>`. Set `RUN_NAME` when you
need a different output name. Keep the BF16 baseline's seed, tokenizer, data
order, batch size, schedule, and evaluation setup matched for comparisons.
