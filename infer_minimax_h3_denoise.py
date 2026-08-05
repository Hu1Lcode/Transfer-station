# -*- coding: utf-8 -*-
"""
MiniMax-H3 静态分组 · 阶段 2：DiT denoise + VAE 解码（world=2，跑在 2,3 两卡）

DiT（transformer）独立部署到 2,3 两卡（FSDP shard=2，每卡 ~31GB bf16），
叠加 Ulysses 序列并行（SP=2，56 头/2=28 可整除）。Qwen3-VL 在阶段 1 已由
infer_minimax_h3_encode.py 在 0,1 卡上完成 prompt encode 并写入 state 文件，
本阶段从 state 恢复 prompt_embeds，跑 denoise loop + VAE 解码，落盘视频+音频。

启动（阶段 2，单机 4 卡中的后 2 卡）：
  # CUDA:
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 infer_minimax_h3_denoise.py \
      --state_path state.pt --num_frames 124 --output out.mp4

  # 昇腾 NPU（装 torch_npu 即可，脚本自动切 npu/hccl）：
  ASCEND_RT_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 infer_minimax_h3_denoise.py \
      --state_path state.pt --num_frames 124 --output out.mp4

两阶段用 run_minimax_h3_2stage.sh 串联。
"""

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist

from diffusers import (
    ContextParallelConfig,
    MiniMaxH3Transformer3DModel,
    ModularPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

from mmh3_common import (
    UlyssesBoundarySharder,
    fsdp_shard_module,
    init_dist_and_mesh,
    xpu,
)


def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 阶段2：DiT denoise + VAE 解码（world=2, 卡2,3）")
    p.add_argument("--model_id", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--state_path", default="state.pt", help="阶段1 产出的 prompt_embeds 文件")
    p.add_argument("--workflow", choices=["t2va", "fl2va", "ref2va"], default="t2va",
                   help="t2va/fl2va 共用 transformer/ 分区；ref2va 加载 transformer_ref/ 分区")
    p.add_argument("--num_frames", type=int, default=124, help="会被吸附到 17*n+5，时长需在 5~15 s")
    p.add_argument("--height", type=int, default=None, help="32 的倍数；不传用模型默认画布")
    p.add_argument("--width", type=int, default=None, help="32 的倍数；不传用模型默认画布")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="minimax_h3_output.mp4")
    p.add_argument("--image", default=None, help="首帧图片路径或 URL（fl2va；应与阶段1一致）")
    p.add_argument("--last_image", default=None, help="尾帧图片路径或 URL（fl2va；应与阶段1一致）")
    p.add_argument("--reference", nargs="*", default=[], help="ref2va 参考文件列表（图/视频/音频，顺序有语义）")
    p.add_argument(
        "--attn_backend",
        default=None,
        help="注意力后端，默认 native(SDPA)；装了 flash-attn 可传 flash，Hopper 可传 _flash_3_hub",
    )
    p.add_argument("--load_stagger", type=float, default=0.0, help="错峰加载，降低 host 内存峰值")
    return p.parse_args()


def build_reference(path):
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        return MiniMaxH3ImageReference.from_file(path)
    if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        return MiniMaxH3VideoReference.from_file(path)
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        return MiniMaxH3AudioReference.from_file(path)
    raise ValueError(f"无法按扩展名识别参考文件类型: {path}")


def main():
    args = parse_args()
    rank, local_rank, world_size, device, fsdp_mesh = init_dist_and_mesh()

    if rank == 0:
        print(f"[denoise] 阶段2：DiT denoise, device={device}, world={world_size}, workflow={args.workflow}", flush=True)

    if args.load_stagger > 0:
        time.sleep(rank * args.load_stagger)

    # 1) 取工作流，pop 掉 text_encoder 子块（阶段1 已 encode，rest 不再需要它）
    workflow = ModularPipeline.from_pretrained(args.model_id).blocks.get_workflow(args.workflow)
    workflow.sub_blocks.pop("text_encoder")
    rest = workflow.init_pipeline(args.model_id)

    # 2) DiT transformer：整份加载到 host -> 逐块 FSDP 分片上卡（shard=2，每卡 ~31GB）
    subfolder = "transformer_ref" if args.workflow == "ref2va" else "transformer"
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model_id, subfolder=subfolder, dtype=torch.bfloat16
    )
    fsdp_shard_module(transformer, transformer.transformer_blocks, fsdp_mesh, device)
    gc.collect()
    xpu.empty_cache()

    # 3) Ulysses SP=2：ulysses_anything 支持任意打包序列长度；cp_plan 传空 dict ——
    #    边界分片由 UlyssesBoundarySharder 处理，这里只要 dispatcher 的 all-to-all
    if args.attn_backend is not None:
        transformer.set_attention_backend(args.attn_backend)
    transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True),
        cp_plan={},
    )
    UlyssesBoundarySharder(transformer, world_size, rank)
    if rank == 0:
        print(f"[denoise] DiT FSDP shard=2 + Ulysses SP=2 完成，显存占用 {xpu.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # 4) 注入 DiT，加载其余共享组件（VAE / 音频 VAE / scheduler），放本阶段 2,3 卡
    rest.update_components(**{subfolder: transformer})
    rest.load_components(workflow=args.workflow, dtype=torch.bfloat16)
    rest.vae.to(device)
    rest.audio_vae.to(device)
    dist.barrier()
    if rank == 0:
        print(f"[denoise] 组件加载完成，显存占用 {xpu.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # 5) 从 state 文件恢复阶段1 的 PipelineState（含 prompt_embeds 等）
    state = torch.load(args.state_path, map_location=device, weights_only=False)

    # 6) 组装请求（与阶段1 的参数一致才可复现；t2va 下 rest 直接从 state 拿 prompt_embeds）
    call_kwargs = dict(
        state=state,
        num_frames=args.num_frames,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    if args.height is not None:
        call_kwargs["height"] = args.height
    if args.width is not None:
        call_kwargs["width"] = args.width
    if args.num_inference_steps is not None:
        call_kwargs["num_inference_steps"] = args.num_inference_steps
    if args.image is not None:
        call_kwargs["image"] = load_image(args.image)
    if args.last_image is not None:
        call_kwargs["last_image"] = load_image(args.last_image)
    if args.reference:
        call_kwargs["references"] = [build_reference(p) for p in args.reference]

    # 7) denoise + VAE 解码：所有 rank 跑完整 pipeline，噪声由相同 seed 保证一致；
    #    transformer 前向内部做 FSDP all-gather + Ulysses all-to-all，
    #    出口收集后各 rank 上的 scheduler / VAE 输入完全相同
    with torch.no_grad():
        results = rest(**call_kwargs)

    # 8) 只在 rank 0 落盘（各 rank 结果一致）
    if rank == 0:
        encode_video(
            results["videos"][0],
            fps=24,
            output_path=args.output,
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
        print(f"[done] 已保存 {args.output}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
