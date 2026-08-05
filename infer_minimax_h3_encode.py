# -*- coding: utf-8 -*-
"""
MiniMax-H3 静态分组 · 阶段 1：Qwen3-VL prompt encode（world=2，跑在 0,1 两卡）

把 Qwen3-VL 文本条件器独立部署到 0,1 两卡（FSDP shard=2，每卡 ~31GB bf16），
只做 prompt encode，产出 prompt_embeds 到 ``--state_path`` 文件，进程即退出——
DiT 从头到尾不加载，无需任何 CPU offload，也不与 DiT 争显存。

阶段 2（denoise）由 infer_minimax_h3_denoise.py 在 2,3 两卡上运行，从 state 文件
恢复 prompt_embeds 继续生成视频+音频。

启动（阶段 1，单机 4 卡中的前 2 卡）：
  # CUDA:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 infer_minimax_h3_encode.py \
      --prompt "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
      --state_path state.pt

  # 昇腾 NPU（装 torch_npu 即可，脚本自动切 npu/hccl）：
  ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 infer_minimax_h3_encode.py \
      --prompt "..." --state_path state.pt

生成约束：24 fps、5~15 s；num_frames 会被吸附到 17*n+5。encode 本身与帧数无关，
但 prompt 相同时两个阶段的请求参数必须一致（seed 等），保证生成可复现。
"""

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist

from diffusers import ModularPipeline
from transformers import Qwen3VLForConditionalGeneration

from mmh3_common import (
    fsdp_shard_module,
    get_qwen3vl_decoder_layers,
    init_dist_and_mesh,
    xpu,
)


def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 阶段1：Qwen3-VL prompt encode（world=2, 卡0,1）")
    p.add_argument("--model_id", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--prompt", required=True)
    p.add_argument("--state_path", default="state.pt", help="prompt_embeds 输出文件，阶段2 从它恢复")
    p.add_argument("--image", default=None, help="首帧图片路径或 URL（fl2va）；encode 阶段若传则写入 state")
    p.add_argument("--last_image", default=None, help="尾帧图片路径或 URL（fl2va）；写入 state")
    p.add_argument("--load_stagger", type=float, default=0.0, help="错峰加载，降低 host 内存峰值")
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size, device, fsdp_mesh = init_dist_and_mesh()

    if rank == 0:
        print(f"[encode] 阶段1：Qwen3-VL prompt encode, device={device}, world={world_size}", flush=True)

    if args.load_stagger > 0:
        time.sleep(rank * args.load_stagger)

    # 1) 取 t2va 工作流，把 text_encoder 子块拆出来单独建 pipeline（官方 recipe 做法）
    workflow = ModularPipeline.from_pretrained(args.model_id).blocks.get_workflow("t2va")
    conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(args.model_id)
    # 只加载 conditioner 自己声明的组件（text_encoder / tokenizer / processor），不碰 transformer/VAE
    conditioner.load_components(workflow="t2va", dtype=torch.bfloat16)

    # 2) Qwen3-VL：整份加载到 host -> 逐块 FSDP 分片上卡（shard=2，每卡 ~31GB）
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", dtype=torch.bfloat16
    )
    fsdp_shard_module(
        text_encoder,
        get_qwen3vl_decoder_layers(text_encoder),
        fsdp_mesh,
        device,
        shard_root=False,
    )
    conditioner.update_components(text_encoder=text_encoder)
    gc.collect()
    xpu.empty_cache()
    if rank == 0:
        print(f"[encode] Qwen3-VL FSDP shard=2 完成，显存占用 {xpu.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # 3) encode：state = conditioner(prompt=...)，返回 PipelineState（含 prompt_embeds）
    encode_kwargs = dict(prompt=args.prompt)
    if args.image is not None:
        from diffusers.utils import load_image

        encode_kwargs["image"] = load_image(args.image)
    if args.last_image is not None:
        from diffusers.utils import load_image

        encode_kwargs["last_image"] = load_image(args.last_image)

    with torch.no_grad():
        state = conditioner(**encode_kwargs)

    # 4) PipelineState 是纯 dataclass（values dict + kwargs_mapping dict），可 pickle。
    #    只在 rank 0 落盘；tensor 走值复制，阶段2 所有 rank 从同一文件恢复即可。
    if rank == 0:
        torch.save(state, args.state_path)
        embeds = state.get("prompt_embeds")
        print(
            f"[encode] 完成，state 已保存到 {args.state_path}，prompt_embeds shape={tuple(embeds.shape) if embeds is not None else None}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
