"""
Deno MiniMax H3 Fast Scene Director (7+1 Architecture)
Ultra-fast multi-scene narrative director designed specifically for the Deno 7+1 workflow.
Chains multiple scenes with Motion Context continuity and 3D latent upscaling.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import folder_paths
import torch

from ..director.core_sampling import sample_single_stage
from ..director.frame_align import minimax_align_frame_count
from ..director.motion_context import apply_exported_motion_context
from ..nodes.conditioning import run_minimax_conditioning

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.fast_director")

CAMERA_PRESETS = [
    "Prompt only",
    "Static / Locked-off",
    "Pan Left",
    "Pan Right",
    "Zoom In / Push in",
    "Zoom Out / Pull back",
    "Tilt Up",
    "Tilt Down",
    "Orbit Left",
    "Orbit Right",
    "Drone Flying Forward",
    "Crane Up",
]

CAMERA_TAGS = {
    "Static / Locked-off": "static camera, locked-off shot, steady frame",
    "Pan Left": "cinematic camera pan left, smooth horizontal pan",
    "Pan Right": "cinematic camera pan right, smooth horizontal pan",
    "Zoom In / Push in": "smooth camera zoom in, push in towards subject",
    "Zoom Out / Pull back": "smooth camera zoom out, wide pull back",
    "Tilt Up": "cinematic camera tilt up towards sky",
    "Tilt Down": "cinematic camera tilt down",
    "Orbit Left": "cinematic 360 orbit camera rotation from left",
    "Orbit Right": "cinematic 360 orbit camera rotation from right",
    "Drone Flying Forward": "aerial drone shot flying forward swiftly",
    "Crane Up": "cinematic crane jib boom shot rising up",
}


def _get_upscaler_model_list():
    models = folder_paths.get_filename_list("latent_upscale_models")
    return list(models) if models else ["minimax_h3_latent_upscaler_3d_bf16.safetensors"]


def _upscale_3d_latent(latent_dict, model_name, scale_factor):
    try:
        from custom_nodes.Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d import (
            MinimaxH3LatentUpscaler3D,
            UpscaleConfig,
            UpscaleMode,
        )
        mode = UpscaleConfig(mode=UpscaleMode.SCALE_BY, scale=float(scale_factor), width=0, height=0, megapixels=0)
        out = MinimaxH3LatentUpscaler3D.execute(
            latent=latent_dict,
            model_name=model_name,
            mode=mode,
            align=16,
            enable_temporal_chunking=True,
            force_unload=True,
            device="cuda",
            precision="bf16",
        )
        if hasattr(out, "outputs"):
            return out.outputs[0]
        elif isinstance(out, (list, tuple)):
            return out[0]
        return out
    except Exception as e:
        log.warning(f"3D Latent Upscaler warning: {e}. Fallback to trilinear interpolation.")
        s = latent_dict["samples"]
        was_5d = (s.dim() == 5)
        if not was_5d:
            s = s.unsqueeze(0)
        b, c, t, h, w = s.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        s_up = torch.nn.functional.interpolate(
            s.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(b, t, c, new_h, new_w).permute(0, 2, 1, 3, 4)
        if not was_5d:
            s_up = s_up.squeeze(0)
        res = latent_dict.copy()
        res["samples"] = s_up
        return res


def _separate_av(latent_dict):
    samples = latent_dict["samples"]
    if samples.shape[0] == 2:
        parts = samples.unbind(0)
        video_l = latent_dict.copy()
        video_l["samples"] = parts[0]
        audio_l = latent_dict.copy()
        audio_l["samples"] = parts[1]
        return video_l, audio_l
    return latent_dict, None


def _concat_av(video_latent, audio_latent):
    if audio_latent is None:
        return video_latent
    res = video_latent.copy()
    res["samples"] = torch.stack([video_latent["samples"], audio_latent["samples"]], dim=0)
    return res


def _split_sigmas_7_1(sigmas):
    if isinstance(sigmas, torch.Tensor) and len(sigmas) >= 8:
        stage1_sigmas = sigmas[:8]
        stage2_sigmas = sigmas[7:]
        return stage1_sigmas, stage2_sigmas
    return sigmas, None


class DenoMiniMaxH3FastDirector:
    """Fast Multi-Scene Director for Deno MiniMax H3 (7+1 Architecture).

    Chains multiple shots/scenes with camera motions, prompt direction, reference
    images, and Motion Context continuity. Powered by low-res 7-step generation,
    3D latent upscaling, and 1-step high-res refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        upscaler_models = _get_upscaler_model_list()
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "MiniMax H3 model (with Turbo LoRA, Sol-Attn, H3 Memory Opt)."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "MiniMax H3 text encoder (Qwen3-VL 32B INT8)."},
                ),
                "video_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 Video VAE (fp16)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 Audio VAE (fp32)."},
                ),
                "sampler": (
                    "SAMPLER",
                    {"tooltip": "KSampler algorithm (e.g. euler)."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Turbo LoRA 8-step sigmas (from BasicScheduler + SigmaShift)."},
                ),
                "latent_upscaler_model": (
                    upscaler_models,
                    {"default": upscaler_models[0], "tooltip": "3D Latent Upscaler model file."},
                ),
                "enable_3d_latent_upscale": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Activate the Deno 7+1 pipeline (7 steps low-res -> 3D Latent Upscale -> 1 step refine)."},
                ),
                "upscale_factor": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 2.5, "step": 0.1, "tooltip": "Multiplier for 3D Latent Upscaler (e.g. 1.5x gives 768x1344 from 512x896)."},
                ),
                "base_width": (
                    "INT",
                    {"default": 512, "min": 384, "max": 1024, "step": 64, "tooltip": "Base generation width before upscale."},
                ),
                "base_height": (
                    "INT",
                    {"default": 896, "min": 384, "max": 1536, "step": 64, "tooltip": "Base generation height before upscale."},
                ),
                "motion_continuity": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable visual & audio Motion Context continuity between consecutive scenes."},
                ),
                "motion_context_frames": (
                    "INT",
                    {"default": 22, "min": 5, "max": 39, "step": 1, "tooltip": "Number of overlap frames for transition continuity (22 frames ~= 0.9s)."},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                # Scene 1
                "scene_1_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Cinematic establishing shot of an ancient futuristic temple hidden in a bioluminescent jungle, mystical atmosphere, 8k photorealistic",
                        "tooltip": "Description of Scene 1.",
                    },
                ),
                "scene_1_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.5, "max": 15.0, "step": 0.5, "tooltip": "Duration of Scene 1 in seconds."},
                ),
                "scene_1_camera": (
                    CAMERA_PRESETS,
                    {"default": "Drone Flying Forward", "tooltip": "Camera motion for Scene 1."},
                ),
                # Scene 2
                "scene_2_enabled": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Enable Scene 2."},
                ),
                "scene_2_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Close-up shot of a glowing mechanical artifact on an ancient altar, intricate gears and light beams, detailed reflections",
                        "tooltip": "Description of Scene 2.",
                    },
                ),
                "scene_2_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.5, "max": 15.0, "step": 0.5, "tooltip": "Duration of Scene 2 in seconds."},
                ),
                "scene_2_camera": (
                    CAMERA_PRESETS,
                    {"default": "Zoom In / Push in", "tooltip": "Camera motion for Scene 2."},
                ),
                # Scene 3
                "scene_3_enabled": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable Scene 3."},
                ),
                "scene_3_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Wide panoramic shot showing the temple activating with giant blue light pillars shooting into the stormy sky, epic scale",
                        "tooltip": "Description of Scene 3.",
                    },
                ),
                "scene_3_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.5, "max": 15.0, "step": 0.5, "tooltip": "Duration of Scene 3 in seconds."},
                ),
                "scene_3_camera": (
                    CAMERA_PRESETS,
                    {"default": "Pan Right", "tooltip": "Camera motion for Scene 3."},
                ),
                # Scene 4
                "scene_4_enabled": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Enable Scene 4."},
                ),
                "scene_4_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Low-angle heroic shot of an explorer observing the phenomenon in awe, dramatic rim lighting",
                        "tooltip": "Description of Scene 4.",
                    },
                ),
                "scene_4_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.5, "max": 15.0, "step": 0.5, "tooltip": "Duration of Scene 4 in seconds."},
                ),
                "scene_4_camera": (
                    CAMERA_PRESETS,
                    {"default": "Tilt Up", "tooltip": "Camera motion for Scene 4."},
                ),
            },
            "optional": {
                "scene_1_image": ("IMAGE", {"tooltip": "Reference image for Scene 1 (Picture 1)."}),
                "scene_2_image": ("IMAGE", {"tooltip": "Reference image for Scene 2 (Picture 1)."}),
                "scene_3_image": ("IMAGE", {"tooltip": "Reference image for Scene 3 (Picture 1)."}),
                "scene_4_image": ("IMAGE", {"tooltip": "Reference image for Scene 4 (Picture 1)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "execute_scenes"
    CATEGORY = "MiniMaxH3/Director"

    def execute_scenes(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
        sampler,
        sigmas,
        latent_upscaler_model: str,
        enable_3d_latent_upscale: bool,
        upscale_factor: float,
        base_width: int,
        base_height: int,
        motion_continuity: bool,
        motion_context_frames: int,
        seed: int,
        cfg: float,
        scene_1_prompt: str,
        scene_1_seconds: float,
        scene_1_camera: str,
        scene_2_enabled: bool,
        scene_2_prompt: str,
        scene_2_seconds: float,
        scene_2_camera: str,
        scene_3_enabled: bool,
        scene_3_prompt: str,
        scene_3_seconds: float,
        scene_3_camera: str,
        scene_4_enabled: bool,
        scene_4_prompt: str,
        scene_4_seconds: float,
        scene_4_camera: str,
        scene_1_image=None,
        scene_2_image=None,
        scene_3_image=None,
        scene_4_image=None,
    ):
        fps = 24.0
        scenes_config = [
            {"enabled": True, "prompt": scene_1_prompt, "seconds": scene_1_seconds, "camera": scene_1_camera, "image": scene_1_image},
            {"enabled": scene_2_enabled, "prompt": scene_2_prompt, "seconds": scene_2_seconds, "camera": scene_2_camera, "image": scene_2_image},
            {"enabled": scene_3_enabled, "prompt": scene_3_prompt, "seconds": scene_3_seconds, "camera": scene_3_camera, "image": scene_3_image},
            {"enabled": scene_4_enabled, "prompt": scene_4_prompt, "seconds": scene_4_seconds, "camera": scene_4_camera, "image": scene_4_image},
        ]

        active_scenes = [s for s in scenes_config if s["enabled"]]
        total_scenes = len(active_scenes)
        log.info(f"[Deno Fast Director] Demarrage du rendu pour {total_scenes} scene(s)...")

        # Sigmas 7+1 setup
        high_sigmas, low_sigmas = _split_sigmas_7_1(sigmas)
        use_7_1 = enable_3d_latent_upscale and (low_sigmas is not None) and (upscale_factor > 1.0)

        accumulated_images = []
        accumulated_audio_waveform = []
        audio_sample_rate = 44100

        prev_context_frames = None
        prev_context_audio = None
        prev_context_latent = None

        for idx, scene in enumerate(active_scenes):
            scene_num = idx + 1
            t_start = time.perf_counter()
            log.info(f"--- [Scene {scene_num}/{total_scenes}] Preparation ---")

            # Format effective prompt with camera motion
            p = scene["prompt"].strip()
            cam = scene["camera"]
            if cam in CAMERA_TAGS:
                p = f"{p}, {CAMERA_TAGS[cam]}"

            ref_images = {}
            if scene["image"] is not None:
                img = scene["image"]
                if img.ndim == 4 and img.shape[0] > 1:
                    img = img[:1]
                ref_images["ref_image_0"] = img
                if "<Picture 1>" not in p:
                    p = f"<Picture 1> {p}"

            # Calculate frame count
            nominal_frames = int(scene["seconds"] * fps)
            context_span = int(motion_context_frames) if (motion_continuity and prev_context_frames is not None) else 0
            generation_request = nominal_frames + context_span
            num_frames = minimax_align_frame_count(generation_request)

            task_key = "r2v" if ref_images else "t2v"
            positive, negative, latent, _ = run_minimax_conditioning(
                clip=clip,
                vae=video_vae,
                audio_vae=audio_vae,
                prompt=p,
                width=base_width,
                height=base_height,
                length=num_frames,
                task_key=task_key,
                ref_images=ref_images if ref_images else None,
            )

            # Apply Motion Context if chaining
            if context_span > 0 and prev_context_frames is not None:
                log.info(f"[Scene {scene_num}] Injection du Motion Context ({context_span} frames d'enchainement)...")
                try:
                    positive, _ = apply_exported_motion_context(
                        positive,
                        video_vae=video_vae,
                        audio_vae=audio_vae,
                        latent=latent,
                        context_frames=prev_context_frames,
                        context_audio=prev_context_audio,
                        context_latent=prev_context_latent,
                        context_span=context_span,
                        target_frame_count=nominal_frames,
                        generation_frame_count=num_frames,
                        visual_enabled=True,
                        audio_enabled=True,
                        fps=fps,
                        task_key=task_key,
                    )
                except Exception as e:
                    log.warning(f"[Scene {scene_num}] Motion Context fallback: {e}")

            # STAGE 1: Fast Sampling (7 steps if 7+1, or full sigmas)
            stage1_sigmas = high_sigmas if use_7_1 else sigmas
            log.info(f"[Scene {scene_num}] Etape 1 : Echantillonnage rapide ({len(stage1_sigmas)-1} steps a {base_width}x{base_height})...")
            sampled_latent = sample_single_stage(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent,
                seed=seed + idx * 100,
                cfg=cfg,
                steps=len(stage1_sigmas) - 1,
                sampler_name="euler",
                scheduler="simple",
                external_sampler=sampler,
                external_sigmas=stage1_sigmas,
            )

            # STAGE 2: 3D Latent Upscaling + 1-step refine
            if use_7_1:
                log.info(f"[Scene {scene_num}] Etape 2 : Upscale 3D Latent direct ({upscale_factor}x) + Raffinement 1-step...")
                video_l, audio_l = _separate_av(sampled_latent)
                upscaled_video = _upscale_3d_latent(video_l, latent_upscaler_model, upscale_factor)
                stage2_in = _concat_av(upscaled_video, audio_l)

                final_latent = sample_single_stage(
                    model=model,
                    positive=positive,
                    negative=negative,
                    latent=stage2_in,
                    seed=seed + idx * 100,
                    cfg=cfg,
                    steps=1,
                    sampler_name="euler",
                    scheduler="simple",
                    external_sampler=sampler,
                    external_sigmas=low_sigmas,
                    denoise=1.0,
                )
            else:
                final_latent = sampled_latent

            # DECODE VAE
            log.info(f"[Scene {scene_num}] Decodage VAE Video & Audio...")
            video_l, audio_l = _separate_av(final_latent)
            v_samples = video_l["samples"]
            if v_samples.ndim == 5 and v_samples.shape[0] == 1:
                v_samples = v_samples.squeeze(0)
            
            decoded_images = video_vae.decode(v_samples)

            # Audio decode
            decoded_audio = None
            if audio_vae is not None and audio_l is not None:
                a_samples = audio_l["samples"]
                if a_samples.ndim == 5 and a_samples.shape[0] == 1:
                    a_samples = a_samples.squeeze(0)
                try:
                    decoded_audio = audio_vae.decode(a_samples)
                except Exception as e:
                    log.warning(f"Audio decode warning: {e}")

            # Store context tail for next scene BEFORE trimming
            if motion_continuity:
                tail_len = min(int(motion_context_frames), int(decoded_images.shape[0]))
                prev_context_frames = decoded_images[-tail_len:].clone()
                prev_context_latent = {"samples": v_samples[:, -((tail_len - 1) // 4 + 1):].clone()}
                prev_context_audio = decoded_audio

            # Trim head overlap frames
            if context_span > 0:
                decoded_images = decoded_images[context_span:]
                if decoded_audio and "waveform" in decoded_audio:
                    # Trim audio samples proportionally
                    wf = decoded_audio["waveform"]
                    sr = decoded_audio.get("sample_rate", 44100)
                    trim_samples = int((context_span / fps) * sr)
                    if wf.shape[-1] > trim_samples:
                        decoded_audio["waveform"] = wf[..., trim_samples:]

            # Cap to nominal frames
            if decoded_images.shape[0] > nominal_frames:
                decoded_images = decoded_images[:nominal_frames]
                if decoded_audio and "waveform" in decoded_audio:
                    sr = decoded_audio.get("sample_rate", 44100)
                    keep_samples = int((nominal_frames / fps) * sr)
                    decoded_audio["waveform"] = decoded_audio["waveform"][..., :keep_samples]

            accumulated_images.append(decoded_images)
            if decoded_audio and "waveform" in decoded_audio:
                accumulated_audio_waveform.append(decoded_audio["waveform"])
                audio_sample_rate = decoded_audio.get("sample_rate", 44100)

            t_elapsed = time.perf_counter() - t_start
            log.info(f"[Scene {scene_num}/{total_scenes}] Terminee en {t_elapsed:.2f}s ({decoded_images.shape[0]} frames).")

        # Concatenate all scenes into final video
        final_images = torch.cat(accumulated_images, dim=0)

        # Concatenate audio
        final_audio = None
        if accumulated_audio_waveform:
            concat_waveform = torch.cat(accumulated_audio_waveform, dim=-1)
            final_audio = {"waveform": concat_waveform, "sample_rate": audio_sample_rate}

        total_frames = final_images.shape[0]
        total_seconds = total_frames / fps
        log.info(f"[Deno Fast Director] SEQUENCE COMPLETE : {total_frames} frames ({total_seconds:.2f}s a 24 fps) rendues avec succes !")

        return (final_images, final_audio, float(fps))
