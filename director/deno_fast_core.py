"""
Helper utilities for Deno 7+1 fast latent upscaler engine integration.
"""

from __future__ import annotations

import logging
import folder_paths
import torch

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.deno_fast_core")

try:
    import comfy_kitchen as ck
    if torch.cuda.is_available() and hasattr(ck, "registry") and not ck.registry.is_available("cuda"):
        ck.registry.enable("cuda")
        log.info("Auto-enabled comfy_kitchen CUDA backend for Sol-Attn.")
except Exception:
    pass


def get_upscaler_model_list() -> list[str]:
    models = folder_paths.get_filename_list("latent_upscale_models")
    return list(models) if models else ["minimax_h3_latent_upscaler_3d_bf16.safetensors"]


def separate_av_latents(latent_dict: dict) -> tuple[dict, dict | None]:
    """Separate the unified nested AV latent into video and audio latent dictionaries."""
    samples = latent_dict.get("samples")
    if samples is not None and isinstance(samples, torch.Tensor) and samples.shape[0] == 2:
        parts = samples.unbind(0)
        video_l = latent_dict.copy()
        video_l["samples"] = parts[0]
        audio_l = latent_dict.copy()
        audio_l["samples"] = parts[1]
        return video_l, audio_l
    return latent_dict, None


def concat_av_latents(video_latent: dict, audio_latent: dict | None) -> dict:
    """Recombine separate video and audio latent dictionaries back into an AV latent."""
    if audio_latent is None:
        return video_latent
    res = video_latent.copy()
    v_s = video_latent["samples"]
    a_s = audio_latent["samples"]
    res["samples"] = torch.stack([v_s, a_s], dim=0)
    return res


def upscale_3d_latent(latent_dict: dict, model_name: str, scale_factor: float) -> dict:
    """Upscale video latent using MinimaxH3LatentUpscaler3D (or bilinear fallback)."""
    if abs(float(scale_factor) - 1.0) < 1e-4:
        return latent_dict
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
        if hasattr(out, "outputs") and out.outputs:
            return out.outputs[0]
        elif isinstance(out, (list, tuple)) and out:
            return out[0]
        return out
    except Exception as e:
        log.warning(f"3D Latent Upscaler warning: {e}. Fallback to spatial interpolation.")
        s = latent_dict["samples"]
        was_5d = (s.dim() == 5)
        if not was_5d:
            s = s.unsqueeze(0)
        b, c, t, h, w = s.shape
        new_h = int(round(h * scale_factor))
        new_w = int(round(w * scale_factor))
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
