# MiniMax-H3 Inference Acceleration via VDE

This example applies Velocity Decomposition and Estimation (VDE) to the official Diffusers MiniMax-H3 ModularPipeline. It keeps separate VDE states for video and audio velocities while sharing the same anchor schedule, because one H3 Transformer forward predicts both modalities.

## Requirements

MiniMax-H3 support is available in Diffusers 0.40.0 and newer. Install the current release or Diffusers from source together with the media dependencies:

```bash
pip install -U "diffusers>=0.40.0" transformers accelerate av
```

The implementation hooks the MiniMax-H3 ModularPipeline denoise block, including a small private Diffusers module. If a future Diffusers release changes that block API, pin a compatible 0.40.x release or update the two denoise-block imports.

The released BF16 Transformer and Qwen3-VL conditioner are very large. CPU offload is enabled by default; expect substantial host RAM usage. See the official MiniMax-H3 Diffusers documentation for int8 and multi-GPU loading recipes.

## How the H3 adaptation works

For a latent state `x` and a Transformer velocity prediction `v`, VDE splits the velocity into a component parallel to `x` and a component normal to `x`:

```text
v = alpha * x + beta * ||x|| * n
```

Here, `n` is the unit direction of the normal component. At a skipped step, the script linearly extrapolates `alpha` and `beta` from the two latest Transformer anchor steps, reuses the latest `n`, and reconstructs the velocity without another Transformer call.

MiniMax-H3 needs a few model-specific details:

- One H3 Transformer call predicts video and audio velocities together, so both streams use the same anchor/skip decision.
- Video and audio use different schedulers and timestep scales. The script therefore keeps two independent VDE histories and extrapolates each stream with its own timestep.
- FL2VA and Ref2VA prepend conditioning rows to the packed sequence. VDE only decomposes and reconstructs generated rows; conditioning rows remain untouched.
- Both official scheduler steps still run at every iteration. Only the expensive Transformer forward is skipped.
- The final iteration is always an anchor so the trajectory ends with a real model prediction.

With the default 50-point schedule, H3 performs 49 Transformer evaluations. `--stable-step 10 --interval 3` reduces this to 24 evaluations; `--interval 2` uses 30. These are call-count reductions, not guaranteed wall-clock speedups, because decoding, offloading, and data transfer are unchanged.

## Text-to-video-and-audio

```bash
# Original 50-step inference
python inference_minimax_h3.py \
  --workflow t2va \
  --prompt "A fox runs through snow while its footsteps crunch loudly."

# Conservative VDE
python inference_minimax_h3.py \
  --workflow t2va \
  --vde --stable-step 10 --interval 2

# Faster VDE starting point
python inference_minimax_h3.py \
  --workflow t2va \
  --vde --stable-step 10 --interval 3
```

## First/last-frame-to-video-and-audio

```bash
python inference_minimax_h3.py \
  --workflow fl2va \
  --image first.png \
  --last-image last.png \
  --vde --stable-step 10 --interval 2
```

Either keyframe may be omitted, but FL2VA requires at least one of them.

## Reference-to-video-and-audio

References are ordered exactly as they appear on the command line. Use `image=`, `video=`, or `audio=` prefixes:

```bash
python inference_minimax_h3.py \
  --workflow ref2va \
  --prompt "The subject follows the reference motion and speaks with the reference voice." \
  --reference image=subject.png \
  --reference video=motion.mp4 \
  --reference audio=voice.wav \
  --vde --stable-step 12 --interval 2
```

Ref2VA requires at least one image or video reference; an audio reference cannot be the only input.

## Options

- `--stable-step`: last zero-based dense warmup step. `10` runs steps 0 through 10 with the full Transformer.
- `--interval`: anchor spacing after warmup. `2` is conservative; `3` is a practical first speed setting.
- `--num-inference-steps`: sigma grid points, including the terminal zero. MiniMax-H3 performs one fewer model evaluation.
- `--attention-backend _flash_3_hub`: optional Hopper attention backend from the official Diffusers recipe.
- `--no-cpu-offload`: keep components on the selected device; only use this when the full pipeline fits.
- `--height` / `--width`: optional multiples of 32. Without them, the official workflow chooses the canvas.

VDE is not recommended for 4-step or 8-step Turbo LoRAs. Validate video quality, audio quality, and audio-video synchronization against the original sampler before increasing the interval.
