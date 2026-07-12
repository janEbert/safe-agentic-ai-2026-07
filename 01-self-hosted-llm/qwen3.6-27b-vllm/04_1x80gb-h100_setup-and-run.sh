#!/bin/bash

if command -v nvidia-smi > /dev/null 2>&1; then
    _world_size="$(nvidia-smi -L | wc -l)"
else
    _world_size=1
fi

_tp_size=1
_max_concurrent_seqs=2

if [ "$_world_size" -lt "$_tp_size" ]; then
    echo 'Not enough GPUs on this machine. Exiting...' > /dev/stderr
    exit 1
fi

_dp_size="$((_world_size / _tp_size))"

docker pull vllm/vllm-openai:latest
docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env VLLM_API_KEY="$VLLM_API_KEY" \
  --env HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest Qwen/Qwen3.6-27B-FP8 \
    --revision 'e89b16ebf1988b3d6befa7de50abc2d76f26eb09' \
    --trust-remote-code \
    --server-model-name Qwen/Qwen3.6-27B \
    --data-parallel-size "$_dp_size" \
    --tensor-parallel-size "$_tp_size" \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --mm-encoder-tp-mode weights \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-num-seqs "$_max_concurrent_seqs" \
    --max-cudagraph-capture-size "$_max_concurrent_seqs" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --host 0.0.0.0 \
    --port 8000
