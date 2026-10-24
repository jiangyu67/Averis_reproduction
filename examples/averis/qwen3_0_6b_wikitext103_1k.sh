#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export MEGATRON_FP4_SIM="${MEGATRON_FP4_SIM:-nvfp4}"
export MEGATRON_FP4_STOCHASTIC_ROUNDING="${MEGATRON_FP4_STOCHASTIC_ROUNDING:-1}"

RUN_NAME="${RUN_NAME:-qwen3-0.6b-${MEGATRON_FP4_SIM}-4gpu-wikitext103-1k}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
/opt/conda/envs/megatron/bin/python -m torch.distributed.run --nproc_per_node=4 pretrain_gpt.py \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /workspace/Qwen3-0.6B-tokenizer \
  --num-layers 28 \
  --hidden-size 1024 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 16 \
  --num-query-groups 8 \
  --max-position-embeddings 40960 \
  --position-embedding-type rope \
  --rotary-base 1000000 \
  --normalization RMSNorm \
  --qk-layernorm \
  --swiglu \
  --disable-bias-linear \
  --seq-length 2048 \
  --micro-batch-size 1 \
  --global-batch-size 12 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --use-distributed-optimizer \
  --transformer-impl local \
  --bf16 \
  --train-iters 1000 \
  --lr 0.0003 \
  --min-lr 0.00003 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.01 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --seed 1234 \
  --log-interval 1 \
  --save "/mnt/llm/checkpoints/${RUN_NAME}" \
  --save-interval 500 \
  --no-rope-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --eval-interval 10 \
  --eval-iters 1 \
  --train-data-path /workspace/wikitext103/train_text_document \
  --valid-data-path /workspace/wikitext103/valid_text_document \
  --test-data-path /workspace/wikitext103/test_text_document \
  --no-masked-softmax-fusion \
  --tensorboard-dir "/workspace/log/${RUN_NAME}" \
  --log-timers-to-tensorboard \
  --log-throughput \
  --log-memory-to-tensorboard \
  --log-validation-ppl-to-tensorboard
