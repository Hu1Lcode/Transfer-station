#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# MiniMax-H3 静态分组两阶段推理串联脚本
#
# 阶段 1（卡 0,1）：Qwen3-VL prompt encode -> state.pt
# 阶段 2（卡 2,3）：DiT denoise + VAE 解码 -> 视频
#
# 用法（CUDA）：
#   bash run_minimax_h3_2stage.sh \
#       --prompt "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
#       --num_frames 124 --output out.mp4
#
# 昇腾 NPU：先 `DEVICE_VIS=ASCEND_RT_VISIBLE_DEVICES` 再跑，脚本自动切 npu/hccl
#   DEVICE_VIS=ASCEND_RT_VISIBLE_DEVICES bash run_minimax_h3_2stage.sh --prompt "..." --output out.mp4
#
# 注意：
#   * --prompt 只喂给阶段 1（encode）；其余参数（--seed/--height/--width/--num_inference_steps/
#     --attn_backend/--image/--last_image/--reference/--output 等）透传给阶段 2（denoise）。
#   * state 文件默认 state.pt，用 --state_path 指定（两阶段共用，且会被阶段 1 覆盖）。
#   * --image/--last_image 建议两阶段都传（阶段 1 需要它 encode 关键帧，阶段 2 需要它做 fl2va 布局）。
set -euo pipefail

DEVICE_VIS="${DEVICE_VIS:-CUDA_VISIBLE_DEVICES}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_SCRIPT="$SCRIPT_DIR/infer_minimax_h3_encode.py"
STAGE2_SCRIPT="$SCRIPT_DIR/infer_minimax_h3_denoise.py"
STATE_PATH="state.pt"

# 把参数拆成：阶段1 的（--prompt/--image/--last_image/--state_path）+ 阶段2 的（其余）
STAGE1_ARGS=()
STAGE2_ARGS=()
HAS_PROMPT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      STAGE1_ARGS+=("$1" "$2"); HAS_PROMPT=1; shift 2 ;;
    --prompt=*)
      STAGE1_ARGS+=("$1"); HAS_PROMPT=1; shift ;;
    --image|--last_image)
      STAGE1_ARGS+=("$1" "$2"); STAGE2_ARGS+=("$1" "$2"); shift 2 ;;
    --image=*|--last_image=*)
      STAGE1_ARGS+=("$1"); STAGE2_ARGS+=("$1"); shift ;;
    --state_path)
      STATE_PATH="$2"; STAGE1_ARGS+=("$1" "$2"); STAGE2_ARGS+=("$1" "$2"); shift 2 ;;
    --state_path=*)
      STATE_PATH="${1#*=}"; STAGE1_ARGS+=("$1"); STAGE2_ARGS+=("$1"); shift ;;
    *)
      STAGE2_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$HAS_PROMPT" -eq 0 ]]; then
  echo "ERROR: --prompt 必填" >&2
  exit 1
fi

echo "=== [stage1] Qwen3-VL prompt encode on cards 0,1 -> $STATE_PATH ==="
$DEVICE_VIS=0,1 torchrun --nproc_per_node=2 "$STAGE1_SCRIPT" \
  "${STAGE1_ARGS[@]}" 2>&1 | tee /tmp/mmh3_stage1.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "阶段1失败，见 /tmp/mmh3_stage1.log" >&2
  exit 1
fi

echo "=== [stage2] DiT denoise + VAE decode on cards 2,3 ==="
$DEVICE_VIS=2,3 torchrun --nproc_per_node=2 "$STAGE2_SCRIPT" \
  "${STAGE2_ARGS[@]}" 2>&1 | tee /tmp/mmh3_stage2.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "阶段2失败，见 /tmp/mmh3_stage2.log" >&2
  exit 1
fi

echo "=== 全部完成 ==="
