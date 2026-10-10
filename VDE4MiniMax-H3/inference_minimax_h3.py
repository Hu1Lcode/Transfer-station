import argparse
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import torch
from diffusers import ComponentsManager
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3Blocks,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (
    MiniMaxH3DenoiseStep,
    MiniMaxH3Ref2VADenoiseStep,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video


class VDEState:
    """Velocity decomposition state for one modality."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.timesteps: Deque[torch.Tensor] = deque(maxlen=2)
        self.alphas: Deque[torch.Tensor] = deque(maxlen=2)
        self.betas: Deque[torch.Tensor] = deque(maxlen=2)
        self.normal_unit: Optional[torch.Tensor] = None

    @staticmethod
    def _scalar_timestep(timestep, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(timestep):
            return timestep.detach().to(device=device, dtype=torch.float32).reshape(())
        return torch.tensor(float(timestep), device=device, dtype=torch.float32)

    @torch.no_grad()
    def observe(self, timestep, latents: torch.Tensor, velocity: torch.Tensor) -> None:
        x = latents.float()
        v = velocity.float()
        if x.shape != v.shape:
            raise ValueError(f"Latent and velocity shapes must match, got {x.shape} and {v.shape}.")

        latent_norm = torch.linalg.vector_norm(x)
        latent_unit = x / latent_norm.clamp_min(self.eps)
        tangential_norm = torch.sum(v * latent_unit)
        normal = v - tangential_norm * latent_unit
        normal_norm = torch.linalg.vector_norm(normal)

        self.timesteps.append(self._scalar_timestep(timestep, x.device))
        self.alphas.append(tangential_norm / latent_norm.clamp_min(self.eps))
        self.betas.append(normal_norm / latent_norm.clamp_min(self.eps))
        self.normal_unit = normal / normal_norm.clamp_min(self.eps)

    @torch.no_grad()
    def estimate(self, timestep, latents: torch.Tensor) -> torch.Tensor:
        if len(self.timesteps) < 2 or self.normal_unit is None:
            raise RuntimeError("VDE needs two anchor observations before estimating a velocity.")

        x = latents.float()
        current_t = self._scalar_timestep(timestep, x.device)
        t1, t2 = self.timesteps
        alpha1, alpha2 = self.alphas
        beta1, beta2 = self.betas
        dt = t2 - t1

        if torch.abs(dt).item() <= self.eps:
            alpha_hat, beta_hat = alpha2, beta2
        else:
            weight = (current_t - t1) / dt
            alpha_hat = torch.lerp(alpha1, alpha2, weight)
            beta_hat = torch.lerp(beta1, beta2, weight)

        latent_norm = torch.linalg.vector_norm(x)
        return alpha_hat * x + beta_hat * latent_norm * self.normal_unit


class _VDEMiniMaxH3DenoiseMixin:
    def __init__(self, stable_step: int = 10, interval: int = 3):
        if stable_step < 1:
            raise ValueError("`stable_step` must be at least 1 so VDE has two warmup anchors.")
        if interval < 1:
            raise ValueError("`interval` must be at least 1.")
        self.stable_step = stable_step
        self.interval = interval
        super().__init__()

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        denoiser = self.sub_blocks["denoiser"]
        scheduler_step = self.sub_blocks["update"]
        total_steps = len(block_state.timesteps)

        if self.stable_step >= total_steps:
            raise ValueError(
                f"`stable_step` ({self.stable_step}) must be smaller than the number of model evaluations "
                f"({total_steps})."
            )

        video_vde = VDEState()
        audio_vde = VDEState()
        transformer_calls = 0

        with self.progress_bar(total=total_steps) as progress_bar:
            for i, video_t in enumerate(block_state.timesteps):
                audio_t = block_state.audio_timesteps[i]
                is_anchor = (
                    i <= self.stable_step
                    or i == total_steps - 1
                    or (i - self.stable_step) % self.interval == 0
                )

                num_condition_video_rows = block_state.num_condition_video_rows
                num_condition_audio_rows = block_state.num_condition_audio_rows
                video_target = block_state.latents[num_condition_video_rows:]
                audio_target = block_state.audio_latents[num_condition_audio_rows:]

                if is_anchor:
                    components, block_state = denoiser(components, block_state, i=i, t=video_t)
                    transformer_calls += 1

                    video_vde.observe(
                        video_t,
                        video_target,
                        block_state.noise_pred[0, num_condition_video_rows:],
                    )
                    audio_vde.observe(
                        audio_t,
                        audio_target,
                        block_state.audio_noise_pred[0, num_condition_audio_rows:],
                    )
                else:
                    video_velocity = video_vde.estimate(video_t, video_target)
                    audio_velocity = audio_vde.estimate(audio_t, audio_target)

                    block_state.noise_pred = torch.zeros(
                        (1, *block_state.latents.shape),
                        device=video_velocity.device,
                        dtype=video_velocity.dtype,
                    )
                    block_state.audio_noise_pred = torch.zeros(
                        (1, *block_state.audio_latents.shape),
                        device=audio_velocity.device,
                        dtype=audio_velocity.dtype,
                    )
                    block_state.noise_pred[0, num_condition_video_rows:] = video_velocity
                    block_state.audio_noise_pred[0, num_condition_audio_rows:] = audio_velocity

                components, block_state = scheduler_step(components, block_state, i=i, t=video_t)
                progress_bar.update()

        self.set_block_state(state, block_state)
        print(f"[INFO] VDE transformer calls: {transformer_calls}/{total_steps}")
        return components, state


class VDEMiniMaxH3DenoiseStep(_VDEMiniMaxH3DenoiseMixin, MiniMaxH3DenoiseStep):
    """VDE denoise loop for T2VA and FL2VA."""


class VDEMiniMaxH3Ref2VADenoiseStep(_VDEMiniMaxH3DenoiseMixin, MiniMaxH3Ref2VADenoiseStep):
    """VDE denoise loop for Ref2VA."""


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def replace_denoise_loop(blocks, replacement) -> None:
    """Replace the H3 loop block without depending on workflow key flattening."""
    for name, block in blocks.sub_blocks.items():
        if isinstance(block, (MiniMaxH3DenoiseStep, MiniMaxH3Ref2VADenoiseStep)):
            blocks.sub_blocks[name] = replacement
            return
        if getattr(block, "sub_blocks", None):
            try:
                replace_denoise_loop(block, replacement)
                return
            except LookupError:
                pass
    raise LookupError("Could not find the MiniMax-H3 denoise loop in the selected workflow.")


def create_pipeline(args):
    blocks = MiniMaxH3Blocks().get_workflow(args.workflow)
    if args.vde:
        if args.workflow == "ref2va":
            vde_denoise = VDEMiniMaxH3Ref2VADenoiseStep(args.stable_step, args.interval)
        else:
            vde_denoise = VDEMiniMaxH3DenoiseStep(args.stable_step, args.interval)
        replace_denoise_loop(blocks, vde_denoise)

    manager = ComponentsManager()
    pipe = blocks.init_pipeline(args.model, components_manager=manager)
    pipe.load_components(dtype=dtype_from_name(args.dtype))

    if args.attention_backend:
        transformer_name = "transformer_ref" if args.workflow == "ref2va" else "transformer"
        getattr(pipe, transformer_name).set_attention_backend(args.attention_backend)

    if args.cpu_offload:
        manager.enable_auto_cpu_offload(
            device=args.device,
            memory_reserve_margin=args.memory_reserve_margin,
        )
    else:
        pipe.to(args.device)
    return pipe


def load_references(specs):
    references = []
    has_visual_reference = False
    factories = {
        "image": MiniMaxH3ImageReference,
        "video": MiniMaxH3VideoReference,
        "audio": MiniMaxH3AudioReference,
    }
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Reference must use TYPE=PATH syntax, got: {spec}")
        kind, path = spec.split("=", 1)
        kind = kind.strip().lower()
        if kind not in factories:
            raise ValueError(f"Reference type must be image, video, or audio, got: {kind}")
        has_visual_reference = has_visual_reference or kind in {"image", "video"}
        references.append(factories[kind].from_file(path))
    if not has_visual_reference:
        raise ValueError("Ref2VA requires at least one image or video reference; audio cannot be used alone.")
    return references


def build_call_kwargs(args):
    kwargs = {
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator().manual_seed(args.seed),
        "output": ["videos", "audio", "sampling_rate"],
    }
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width

    if args.workflow == "fl2va":
        if args.image is None and args.last_image is None:
            raise ValueError("FL2VA requires `--image`, `--last-image`, or both.")
        if args.image is not None:
            kwargs["image"] = load_image(args.image)
        if args.last_image is not None:
            kwargs["last_image"] = load_image(args.last_image)
    elif args.workflow == "ref2va":
        if not args.reference:
            raise ValueError("Ref2VA requires at least one `--reference TYPE=PATH` argument.")
        kwargs["references"] = load_references(args.reference)
    return kwargs


def get_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3 inference with optional dual-modality VDE acceleration.")
    parser.add_argument("--model", type=str, default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--workflow", choices=["t2va", "fl2va", "ref2va"], default="t2va")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A red fox trots through a snowy pine forest while snow crunches underfoot.",
    )
    parser.add_argument("--image", type=str, default=None, help="First keyframe for FL2VA.")
    parser.add_argument("--last-image", type=str, default=None, help="Last keyframe for FL2VA.")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="TYPE=PATH",
        help="Ordered Ref2VA input. Repeat with image=..., video=..., or audio=....",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--no-cpu-offload", dest="cpu_offload", action="store_false")
    parser.set_defaults(cpu_offload=True)
    parser.add_argument("--memory-reserve-margin", type=str, default="12GB")
    parser.add_argument("--vde", action="store_true", help="Enable dual video/audio VDE acceleration.")
    parser.add_argument("--stable-step", type=int, default=10)
    parser.add_argument("--interval", type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    pipe = create_pipeline(args)
    call_kwargs = build_call_kwargs(args)

    start = time.perf_counter()
    result = pipe(**call_kwargs)
    elapsed = time.perf_counter() - start

    tag = f"vde_s{args.stable_step}i{args.interval}" if args.vde else "original"
    output = Path(args.output or f"output_minimax_h3_{args.workflow}_{tag}_seed{args.seed}.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        result["videos"][0],
        fps=24,
        output_path=str(output),
        audio=result["audio"][0],
        audio_sample_rate=result["sampling_rate"],
    )

    print(f"[INFO] workflow: {args.workflow}")
    print(f"[INFO] method: {tag}")
    print(f"[INFO] saved: {output}")
    print(f"[INFO] elapsed time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
