# -*- coding: utf-8 -*-
"""
MiniMax-H3 四卡推理脚本：FSDP2（权重分片） + Ulysses 序列并行（SP=4）

模型：MiniMaxAI/MiniMax-H3 —— 视频 + 音频联合生成（24 fps，5~15 s）。
  - transformer 本体约 61.7 GB（bf16），Qwen3-VL 文本条件器约 62.1 GB（bf16），
    单卡放不下 -> 用 FSDP2 ``fully_shard`` 把两者按 block 分片到 4 张卡，
    每张卡只常驻 1/4 权重（各约 15.5 GB）。
  - transformer 的注意力走 Ulysses SP：打包序列（text / 条件 / 视频 / 音频行）
    在 forward 入口按 rank 连续切成 4 段，attention 内部由 diffusers dispatcher
    做 all-to-all（gather seq / scatter heads -> 本地注意力 -> 反向 all-to-all），
    forward 出口再把各 rank 的输出行 all-gather 回完整结果。
    因为打包序列是用 video_indices / audio_indices / text_indices 散射组装的，
    diffusers 通用的声明式 ``_cp_plan`` 无法表达（需要按索引重映射），所以入口
    分片 / 出口收集由本脚本的 ``UlyssesBoundarySharder`` 以前向 hook 实现，
    attention 内部仍复用官方 ``enable_parallelism`` + dispatcher 的 Ulysses 实现。

依赖：
  pip install "git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head"
  pip install -U "transformers" accelerate safetensors av
  # 可选：flash-attn（装好后用 --attn_backend flash 提速，SDPA 后端无需安装）

启动（单机 4 卡）：
  # t2va：文本 -> 视频+音频
  torchrun --nproc_per_node=4 infer_minimax_h3_fsdp_ulysses.py \
      --prompt "A red fox trotting through a snowy pine forest, snow crunching underfoot" \
      --num_frames 124 --output minimax_h3_t2va.mp4

  # 若 Qwen3-VL 占显存太多，加 --offload_text_encoder 把它的 sharded 权重常驻 CPU，
  # 只在 prompt encode 时临时 H2D 上卡，编码完即 reshard 回 CPU，腾出显存给 transformer：
  torchrun --nproc_per_node=4 infer_minimax_h3_fsdp_ulysses.py \
      --prompt "..." --num_frames 124 --offload_text_encoder --output out.mp4

  # fl2va：首帧（可含尾帧）引导，与 t2va 共用同一份 transformer 权重
  torchrun --nproc_per_node=4 infer_minimax_h3_fsdp_ulysses.py \
      --prompt "..." --image keyframe.jpg --num_frames 124 --output out.mp4

  # ref2va：参考图/视频/音频引导（加载 transformer_ref 分区）
  torchrun --nproc_per_node=4 infer_minimax_h3_fsdp_ulysses.py \
      --workflow ref2va --prompt "..." \
      --reference subject.jpg motion.mp4 voice.wav --num_frames 124 --output out.mp4

注意：
  * 每个 rank 加载时会在 host 内存短暂物化整份权重（瞬时峰值约 62 GB/rank），
    host 内存紧张时用 --load_stagger 60 让各 rank 错峰加载。
  * 推荐 4 x 80 GB 卡；两个大组件分片后各占约 15.5 GB 显存/rank。
  * 生成约束：24 fps、5~15 s，num_frames 会被吸附到 17*n+5；height/width 需为
    32 的倍数，不传则用模型默认画布（960x544 比 1344x768 每步快约 2.3x）。
"""

import argparse
import gc
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, OffloadPolicy

from diffusers import (
    ContextParallelConfig,
    MiniMaxH3Transformer3DModel,
    ModularPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from transformers import Qwen3VLForConditionalGeneration


# ---------------------------------------------------------------------------
# FSDP2 权重分片
# ---------------------------------------------------------------------------
def fsdp_shard_module(module, blocks, mesh, device, shard_root=True, offload_policy=None):
    """按 block 分片 module：逐块搬上 GPU 后 fully_shard，瞬时只占一块整权重。

    FSDP2 在前向时逐块 all-gather 出完整权重、用完即 reshard，因此显存峰值约为
    “全量 1/world_size + 1~2 个完整 block”。

    offload_policy: 传 CPUOffloadPolicy() 时，sharded 权重常驻 CPU，forward 前 H2D
        all-gather 上卡、用完 reshard 回 CPU，把权重显存腾空。用于 Qwen3-VL——它只在
        denoise loop 前做一次 prompt encode，之后再不被调用，offload 后能给 transformer
        让出权重等量的显存。
    """
    module.requires_grad_(False)
    fsdp_kwargs = dict(mesh=mesh)
    if offload_policy is not None:
        fsdp_kwargs["offload_policy"] = offload_policy

    # 先把整个 module 搬上卡：FSDP2 的 DTensor sharding 要求参数在目标 device 上初始化，
    # 才能按 mesh 切分。CPU offload policy 在 fully_shard 之后自行管理 sharded 参数的
    # H2D/D2H（forward 前 all-gather 上卡、用完 reshard 回 CPU），我们不再手动 .to()。
    module.to(device)
    for block in blocks:
        fully_shard(block, **fsdp_kwargs)
    if shard_root:
        # 根模块再 shard 一次，覆盖不属于任何 block 的参数（embed/norm/head 等）
        fully_shard(module, **fsdp_kwargs)
    return module


def get_qwen3vl_decoder_layers(text_encoder):
    """兼容不同 transformers 版本，定位 Qwen3-VL 语言模型的 decoder layer 列表。"""
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.language_model.layers",
    ]
    for path in candidates:
        obj = text_encoder
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
            return obj
    raise RuntimeError(
        "无法定位 Qwen3-VL 的 decoder layers（尝试过: "
        + ", ".join(candidates)
        + "），请检查 transformers 版本。"
    )


# ---------------------------------------------------------------------------
# Ulysses SP：transformer 边界的序列分片 / 输出收集
# ---------------------------------------------------------------------------
class UlyssesBoundarySharder:
    """给 MiniMaxH3Transformer3DModel 挂上入口分片 / 出口收集 hook。

    打包序列 [0, S) 按 rank 连续切成 world_size 段（与 dispatcher 内 all-to-all
    重组顺序一致，采用 tensor_split 式均衡切分，配合 ulysses_anything 支持 S 不
    整除 world_size 的情况）。每个 rank 只保留落在本段内的视频/音频/文本行，
    并把索引重映射为段内局部坐标；出口把各 rank 算出的行按全局行号 all-gather
    拼回完整结果，保证下游 scheduler / VAE 在所有 rank 上看到完全相同的张量。
    """

    def __init__(self, transformer, world_size, rank, group=None):
        self.world_size = world_size
        self.rank = rank
        self.group = group  # None -> 默认 WORLD 组（ulysses_degree == world_size 时与 ulysses 组相同）
        self._ctx = None
        transformer.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        transformer.register_forward_hook(self._post_hook)

    @staticmethod
    def _shard_bounds(seq_len, world_size, rank):
        base, rem = divmod(seq_len, world_size)
        start = rank * base + min(rank, rem)
        end = start + base + (1 if rank < rem else 0)
        return start, end

    def _pre_hook(self, module, args, kwargs):
        if args:
            raise RuntimeError(
                "UlyssesBoundarySharder 要求 transformer 以关键字参数调用（pipeline 内部即如此）。"
            )

        seq_len = kwargs["position_ids"].shape[0]
        start, end = self._shard_bounds(seq_len, self.world_size, self.rank)

        ctx = {
            "bounds": [self._shard_bounds(seq_len, self.world_size, r) for r in range(self.world_size)],
            # 完整的行索引在所有 rank 上相同，出口收集时要用它推算每个 rank 持有的全局行号
            "video_indices_full": kwargs["video_indices"],
            "audio_indices_full": kwargs["audio_indices"],
        }

        # 逐行结构参数：直接按段切片
        for name in ("position_ids", "token_tags", "timestep_indices"):
            kwargs[name] = kwargs[name][start:end]

        # 各模态行：保留落在本段内的行，索引重映射为段内局部坐标
        for rows_name, idx_name in (
            ("hidden_states", "video_indices"),
            ("audio_hidden_states", "audio_indices"),
            ("encoder_hidden_states", "text_indices"),
        ):
            indices = kwargs[idx_name]
            keep = (indices >= start) & (indices < end)
            row_ids = keep.nonzero(as_tuple=True)[0]
            kwargs[rows_name] = kwargs[rows_name].index_select(1, row_ids)
            kwargs[idx_name] = indices[keep] - start

        # timestep 是“去重后的噪声层级表”，按 timestep_indices 索引，整表保留
        self._ctx = ctx
        return args, kwargs

    def _post_hook(self, module, args, output):
        ctx = self._ctx
        self._ctx = None
        if ctx is None:
            return output

        if hasattr(output, "sample") and hasattr(output, "audio_sample"):
            video = self._gather_rows(output.sample, ctx["video_indices_full"], ctx["bounds"])
            audio = self._gather_rows(output.audio_sample, ctx["audio_indices_full"], ctx["bounds"])
            return type(output)(sample=video, audio_sample=audio)

        video = self._gather_rows(output[0], ctx["video_indices_full"], ctx["bounds"])
        audio = self._gather_rows(output[1], ctx["audio_indices_full"], ctx["bounds"])
        return (video, audio)

    def _gather_rows(self, local_out, indices_full, bounds):
        """local_out: (B, n_local, C)，行为本段内的模态行 -> 拼回 (B, n_total, C)。

        各 rank 的分段边界与完整索引在所有 rank 上一致，因此每个 rank 都能无通信地
        推算出所有 rank 持有的全局行号，all_gather 后原地拼回即可。
        """
        total_rows = indices_full.shape[0]
        if total_rows == 0:
            return local_out

        rows_per_rank, counts = [], []
        for r in range(self.world_size):
            s, e = bounds[r]
            rows = ((indices_full >= s) & (indices_full < e)).nonzero(as_tuple=True)[0]
            rows_per_rank.append(rows)
            counts.append(rows.numel())

        n_local = local_out.shape[1]
        assert n_local == counts[self.rank], (
            f"rank {self.rank}: 本地输出行数 {n_local} 与入口分片数 {counts[self.rank]} 不一致"
        )

        max_count = max(counts)
        send = local_out.new_zeros((local_out.shape[0], max_count, local_out.shape[2]))
        send[:, :n_local] = local_out
        recv = [torch.empty_like(send) for _ in range(self.world_size)]
        dist.all_gather(recv, send.contiguous(), group=self.group)

        full = local_out.new_zeros((local_out.shape[0], total_rows, local_out.shape[2]))
        for r, rows in enumerate(rows_per_rank):
            if counts[r] > 0:
                full[:, rows] = recv[r][:, : counts[r]]
        return full


# ---------------------------------------------------------------------------
# 参考输入（ref2va）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MiniMax-H3 4-GPU inference with FSDP2 + Ulysses SP")
    p.add_argument("--model_id", default="MiniMaxAI/MiniMax-H3")
    p.add_argument(
        "--workflow",
        choices=["t2va", "fl2va", "ref2va"],
        default="t2va",
        help="t2va/fl2va 共用 transformer/ 分区（传 --image 即为 fl2va）；ref2va 加载 transformer_ref/ 分区",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--image", default=None, help="首帧图片路径或 URL（触发 fl2va）")
    p.add_argument("--last_image", default=None, help="尾帧图片路径或 URL（fl2va）")
    p.add_argument("--reference", nargs="*", default=[], help="ref2va 参考文件列表（图/视频/音频，顺序有语义）")
    p.add_argument("--num_frames", type=int, default=124, help="会被吸附到 17*n+5，时长需在 5~15 s")
    p.add_argument("--height", type=int, default=None, help="32 的倍数；不传用模型默认画布")
    p.add_argument("--width", type=int, default=None, help="32 的倍数；不传用模型默认画布")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="minimax_h3_output.mp4")
    p.add_argument(
        "--attn_backend",
        default=None,
        help="注意力后端，默认 native(SDPA，免装额外依赖)；装了 flash-attn 可传 flash，Hopper 可传 _flash_3_hub",
    )
    p.add_argument(
        "--load_stagger",
        type=float,
        default=0.0,
        help="每个 rank 加载大权重前按 rank*stagger 秒错峰，降低 host 内存瞬时峰值",
    )
    p.add_argument(
        "--offload_text_encoder",
        action="store_true",
        help="把 Qwen3-VL 文本条件器的 sharded 权重常驻 CPU RAM——它只在 denoise loop 前做一次 "
        "prompt encode，用完即腾空显存，给 transformer 让出约 1/world_size × 62 GB 的显存",
    )
    return p.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # FSDP 分片数与 Ulysses degree 都取 world_size（典型 4 卡；56 个头可被 1/2/4/7/8 整除，
    # 其它卡数依赖 ulysses_anything 的头部 padding，也能跑但略有开销）
    if world_size < 2:
        raise RuntimeError("请用 torchrun 以多卡启动，例如 torchrun --nproc_per_node=4 ...")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    fsdp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))

    if rank == 0:
        print(f"[setup] world_size={world_size}, device={device}, workflow={args.workflow}", flush=True)

    # 1) 建 pipeline：不带 workflow，只拉取配置，按调用时的输入自动选择工作流
    pipe = ModularPipeline.from_pretrained(args.model_id)

    if args.load_stagger > 0:
        time.sleep(rank * args.load_stagger)

    # 2) transformer：整份加载到 host -> 逐块 FSDP 分片上卡
    subfolder = "transformer_ref" if args.workflow == "ref2va" else "transformer"
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model_id, subfolder=subfolder, dtype=torch.bfloat16
    )
    fsdp_shard_module(transformer, transformer.transformer_blocks, fsdp_mesh, device)
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Ulysses SP：ulysses_anything 支持任意打包序列长度；cp_plan 传空 dict —
    #    边界分片由 UlyssesBoundarySharder 处理，这里只要 dispatcher 的 all-to-all
    if args.attn_backend is not None:
        transformer.set_attention_backend(args.attn_backend)
    transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True),
        cp_plan={},
    )
    UlyssesBoundarySharder(transformer, world_size, rank)

    # 4) Qwen3-VL 文本条件器：decoder layer 逐块分片；embed/visual/lm_head 保持复制。
    #    --offload_text_encoder 时给 sharded 参数挂 CPUOffloadPolicy：常驻 CPU RAM，
    #    prompt encode 时逐层 H2D all-gather 上卡、用完 reshard 回 CPU，
    #    把约 1/world_size × 62 GB 显存腾给后续 transformer denoise loop。
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", dtype=torch.bfloat16
    )
    text_encoder_offload = CPUOffloadPolicy() if args.offload_text_encoder else None
    fsdp_shard_module(
        text_encoder,
        get_qwen3vl_decoder_layers(text_encoder),
        fsdp_mesh,
        device,
        shard_root=False,
        offload_policy=text_encoder_offload,
    )
    if rank == 0:
        print(
            f"[setup] text_encoder offload: "
            f"{'CPU (sharded 权重常驻 CPU, prompt encode 时按需 H2D)' if text_encoder_offload else 'no (常驻 GPU)'}",
            flush=True,
        )
    gc.collect()
    torch.cuda.empty_cache()

    # 5) 注入已分片的组件，再加载其余共享组件（VAE / 音频 VAE / tokenizer / processor / scheduler）
    pipe.update_components(**{subfolder: transformer, "text_encoder": text_encoder})
    pipe.load_components(workflow="t2va" if args.workflow != "ref2va" else "ref2va", dtype=torch.bfloat16)
    pipe.vae.to(device)
    pipe.audio_vae.to(device)
    dist.barrier()
    if rank == 0:
        print(f"[setup] 加载完成，rank0 显存占用 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    # 6) 组装请求
    call_kwargs = dict(
        prompt=args.prompt,
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

    # 7) 推理：所有 rank 跑完整 pipeline，噪声由相同 seed 保证一致；
    #    transformer 前向内部做 FSDP all-gather + Ulysses all-to-all，
    #    出口收集后各 rank 上的 scheduler / VAE 输入完全相同
    with torch.no_grad():
        results = pipe(**call_kwargs)

    # 8) 只在 rank0 落盘（各 rank 结果一致）
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
