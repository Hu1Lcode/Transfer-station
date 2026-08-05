# -*- coding: utf-8 -*-
"""MiniMax-H3 推理公共模块：设备自适应 + FSDP2 分片 + Ulysses 边界分片。

供两个独立进程组脚本共用：
  - infer_minimax_h3_encode.py   阶段 1：Qwen3-VL prompt encode（world=2，跑在 0,1）
  - infer_minimax_h3_denoise.py  阶段 2：DiT denoise + VAE 解码（world=2，跑在 2,3）

静态分组设计：Qwen3-VL 常驻 0,1 两卡（FSDP shard=2），DiT 常驻 2,3 两卡
（FSDP shard=2 + Ulysses SP=2）。两组进程互不争显存，因此不需要任何 CPU offload。
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, OffloadPolicy

# --- 设备后端自适应：有 torch_npu 走昇腾（npu/hccl），否则走 CUDA/cuda/nccl ---
try:
    import torch_npu  # noqa: F401  注册 npu 设备，必须在用 npu 之前 import
    DEVICE = "npu"
    DIST_BACKEND = "hccl"
    xpu = torch.npu
except ImportError:
    DEVICE = "cuda"
    DIST_BACKEND = "nccl"
    xpu = torch.cuda


def init_dist_and_mesh():
    """初始化进程组与 FSDP device mesh，返回 (rank, local_rank, world_size, device, fsdp_mesh)。

    每个脚本都以 ``torchrun --nproc_per_node=2`` 启动，world=2：
    mesh 是干净的 2-rank 一维 mesh，FSDP shard=2 天然成立。
    """
    rank = int(dist.get_rank())
    local_rank = int(dist.get_local_rank())
    world_size = dist.get_world_size()
    assert world_size == 2, f"静态分组脚本按 world=2 设计（每阶段 2 卡），当前 world_size={world_size}"

    xpu.set_device(local_rank)
    device = torch.device(DEVICE, local_rank)
    fsdp_mesh = init_device_mesh(DEVICE, (world_size,), mesh_dim_names=("fsdp",))
    return rank, local_rank, world_size, device, fsdp_mesh


def fsdp_shard_module(module, blocks, mesh, device, shard_root=True, offload_policy=None):
    """按 block 分片 module：逐块搬上设备后 fully_shard，瞬时只占一块整权重。

    FSDP2 在前向时逐块 all-gather 出完整权重、用完即 reshard，因此显存峰值约为
    “全量 1/world_size + 1~2 个完整 block”。

    offload_policy: 传 CPUOffloadPolicy() 时，sharded 参数必须 materialize 在 CPU
    （_validate_cpu_offload_params 会检查 sharded_param.device.type == "cpu"），
    所以 offload 路径不 .to(device)，fully_shard 后 sharded_param 即落 CPU、forward
    时才 H2D。非 offload 路径相反：逐块 .to(device) 再 shard，避免前向时 CPU->卡拷贝。
    """
    module.requires_grad_(False)
    fsdp_kwargs = dict(mesh=mesh)
    if offload_policy is not None:
        fsdp_kwargs["offload_policy"] = offload_policy

    if offload_policy is not None:
        # CPU offload 路径：参数全程留 CPU（from_pretrained 默认加载到 CPU），绝不 .to(device)。
        for block in blocks:
            fully_shard(block, **fsdp_kwargs)
        if shard_root:
            fully_shard(module, **fsdp_kwargs)
    else:
        # 非 offload 路径：逐块上卡 + 立即 fully_shard，单卡峰值 = 一个 block 的大小，
        # 而不是整个 module（transformer 61.7GB / Qwen3-VL 62GB 全量上单卡必爆显存）。
        for block in blocks:
            block.to(device)
            fully_shard(block, **fsdp_kwargs)
        module.to(device)  # 其余未分片参数（embed/norm/head 等）整体上卡
        if shard_root:
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


# 模块内 re-export（与脚本内用法对齐）
__all__ = [
    "DEVICE",
    "DIST_BACKEND",
    "xpu",
    "init_dist_and_mesh",
    "fsdp_shard_module",
    "get_qwen3vl_decoder_layers",
    "UlyssesBoundarySharder",
]
