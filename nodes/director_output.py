"""Public output contract for MiniMax H3 Motion Director."""

from __future__ import annotations

from .director_inputs import MiniMaxH3MotionDirector as _UnifiedDirector


class MiniMaxH3MotionDirector(_UnifiedDirector):
    """Unified Director with optional graph outputs and self-executing output-node behavior."""

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_NODE = True

    DESCRIPTION = (
        "MiniMax H3 Motion Director. Runs as an OUTPUT_NODE when used standalone, "
        "while still exposing images / audio / fps for optional downstream ComfyUI processing."
    )

    def execute(self, *args, **kwargs):
        # source_images is no longer a public output. Prevent the inherited
        # implementation from building an unused source-image batch even if an
        # old workflow still contains export_source_images=True.
        kwargs["export_source_images"] = False

        outputs = super().execute(*args, **kwargs)
        return outputs[0], outputs[1], outputs[2]


class DenoMiniMaxH3MotionDirector(_UnifiedDirector):
    """Unified Deno 7+1 Motion Director with 'Open Director' timeline modal and fast latent upscaler engine."""

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Deno MiniMax H3 Motion Director (7+1 Fast). Copie complète du Motion Director "
        "avec bouton 'Open Director', panneau de montage visuel timeline et moteur accéléré 7+1."
    )

    def execute(self, *args, **kwargs):
        kwargs["export_source_images"] = False
        kwargs["fast_engine_7_1"] = kwargs.get("fast_engine_7_1", True)
        kwargs["latent_upscaler_model"] = kwargs.get("latent_upscaler_model", "minimax_h3_latent_upscaler_3d_bf16.safetensors")
        kwargs["upscale_factor"] = float(kwargs.get("upscale_factor", 1.5))

        outputs = super().execute(*args, **kwargs)
        return outputs[0], outputs[1], outputs[2]
