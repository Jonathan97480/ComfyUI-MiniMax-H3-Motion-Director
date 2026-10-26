# MiniMax H3 AI Story Director Node
# Author: Jonathan97480 & Contributors
# Licensed under GNU GPL v3.0

"""ComfyUI Node for AI Storyboarding and Narrative Video Planning with MiniMax H3 Motion Director."""

from __future__ import annotations

import json
import logging
from typing import Any

import torch

try:
    from ..director.storyboard import (
        ASPECT_RATIOS,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_STYLE_PROMPT,
        MMX_STORYBOARD,
        StoryboardData,
        StoryboardSubject,
        build_director_timeline,
        format_script_summary,
        generate_llm_scenes,
        generate_llm_storyboard,
        generate_local_scenes,
        generate_local_storyboard,
        generate_scene_reference_images,
        generate_subject_reference_images,
        parse_aspect_ratio,
    )
except (ImportError, ValueError):
    from director.storyboard import (
        ASPECT_RATIOS,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_STYLE_PROMPT,
        MMX_STORYBOARD,
        StoryboardData,
        StoryboardSubject,
        build_director_timeline,
        format_script_summary,
        generate_llm_scenes,
        generate_llm_storyboard,
        generate_local_scenes,
        generate_local_storyboard,
        generate_scene_reference_images,
        generate_subject_reference_images,
        parse_aspect_ratio,
    )

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.storyboard")


class MiniMaxH3StoryDirector:
    """AI Storyboard & Narrative Director for MiniMax H3.

    Transforms a user's story script into a multi-shot cinematic timeline,
    generates per-scene reference images, and automatically populates the
    MiniMax H3 Motion Director.
    """

    @classmethod
    def INPUT_TYPES(cls):
        aspect_keys = list(ASPECT_RATIOS.keys())
        return {
            "required": {
                "story": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Un chevalier cyberpunk explore les ruelles d'une ville néon sous la pluie battante. "
                            "Il rencontre un renard holographique lumineux qui le guide vers une porte secrète dans un temple antique futuriste."
                        ),
                        "tooltip": "Décrivez l'histoire, la scène ou le script que vous souhaitez transformer en vidéo multi-plans.",
                    },
                ),
                "total_duration_sec": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 2.0,
                        "max": 300.0,
                        "step": 1.0,
                        "tooltip": "Durée totale souhaitée pour la vidéo en secondes (ex: 15s, 30s, 60s).",
                    },
                ),
                "shot_duration_sec": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 2.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "Durée cible de chaque scène/plan (MiniMax excelle sur des plans de 5 à 6 secondes).",
                    },
                ),
                "aspect_ratio": (
                    aspect_keys,
                    {
                        "default": "16:9 (864x480)",
                        "tooltip": "Format d'image et résolution des plans vidéo.",
                    },
                ),
                "director_mode": (
                    ["mixed", "fl2v", "r2v", "t2v"],
                    {
                        "default": "mixed",
                        "tooltip": "Mode du Director. 'mixed' permet d'enchaîner plusieurs scènes avec plans, mouvements de caméra et références distincts.",
                    },
                ),
                "style_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_STYLE_PROMPT,
                        "tooltip": "Style visuel, rendu, photographie et direction artistique appliqués à chaque scène.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_NEGATIVE_PROMPT,
                        "tooltip": "Prompt négatif pour la génération d'images de référence.",
                    },
                ),
                "ai_story_engine": (
                    [
                        "Built-in AI Story Engine (Local / No Server needed)",
                        "Ollama (Local LLM)",
                        "OpenAI-Compatible (LM Studio / Jan / LocalAI)",
                    ],
                    {
                        "default": "Built-in AI Story Engine (Local / No Server needed)",
                        "tooltip": "Moteur d'IA narrative pour découper l'histoire en plans cinématographiques.",
                    },
                ),
                "llm_url": (
                    "STRING",
                    {
                        "default": "http://127.0.0.1:11434",
                        "tooltip": "URL du serveur local (ex: Ollama http://127.0.0.1:11434 ou LM Studio http://127.0.0.1:1234/v1).",
                    },
                ),
                "llm_model": (
                    "STRING",
                    {
                        "default": "qwen2.5:7b",
                        "tooltip": "Nom du modèle local (ex: qwen2.5:7b, gemma2:2b, mistral, llama3, etc.).",
                    },
                ),
                "image_steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Nombre d'étapes d'échantillonnage pour les images de référence générées.",
                    },
                ),
                "image_cfg": (
                    "FLOAT",
                    {
                        "default": 6.5,
                        "min": 0.5,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Guidance scale (CFG) pour les images de référence.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Graine aléatoire pour la génération d'images et variations.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Optionnel: Connectez un modèle de diffusion pour générer de vraies images de référence IA pour chaque scène.",
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "Optionnel: Encodeur CLIP associé au modèle de diffusion.",
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": "Optionnel: VAE pour décoder les images de référence générées.",
                    },
                ),
                "external_images": (
                    "IMAGE",
                    {
                        "tooltip": "Optionnel: Lot d'images pré-existantes à utiliser comme références de scènes.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (MMX_STORYBOARD, "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("storyboard", "timeline_json", "reference_images", "script_summary")
    FUNCTION = "execute"
    CATEGORY = "MiniMaxH3/Director"
    DESCRIPTION = (
        "MiniMax H3 AI Story Director. Transforme une histoire narrative en timeline cinématographique "
        "multi-plans avec prompts, mouvements de caméra, durées et images de référence générées automatiquement. "
        "Connectez la sortie 'storyboard' à l'entrée 'storyboard' du MiniMax H3 Motion Director."
    )

    def execute(
        self,
        story: str,
        total_duration_sec: float,
        shot_duration_sec: float,
        aspect_ratio: str,
        director_mode: str,
        style_prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        ai_story_engine: str = "Built-in AI Story Engine (Local / No Server needed)",
        llm_url: str = "http://127.0.0.1:11434",
        llm_model: str = "qwen2.5:7b",
        image_steps: int = 20,
        image_cfg: float = 6.5,
        seed: int = 0,
        control_after_generate: str = "fixed",
        model=None,
        clip=None,
        vae=None,
        external_images: torch.Tensor | None = None,
        unique_id: str | None = None,
        **kwargs,
    ):
        # Defensive type coercion for robustness against any UI widget mismatch
        try:
            target_duration = max(2.0, float(total_duration_sec))
        except (ValueError, TypeError):
            target_duration = 15.0

        try:
            target_shot = max(1.5, float(shot_duration_sec))
        except (ValueError, TypeError):
            target_shot = 5.0

        try:
            steps_val = max(1, int(image_steps))
        except (ValueError, TypeError):
            steps_val = 20

        try:
            cfg_val = max(0.1, float(image_cfg))
        except (ValueError, TypeError):
            cfg_val = 6.5

        try:
            seed_val = int(seed)
        except (ValueError, TypeError):
            seed_val = 0

        neg_prompt_str = str(negative_prompt or DEFAULT_NEGATIVE_PROMPT)
        num_scenes = max(1, int(round(target_duration / target_shot)))
        width, height = parse_aspect_ratio(aspect_ratio)

        log.info(
            "MiniMax H3 Story Director: Generating storyboard for %d scenes (total %.1fs, %dx%d)",
            num_scenes,
            target_duration,
            width,
            height,
        )

        # 1. Narrative decomposition and multi-subject structured generation
        sb_data = None
        engine_lower = str(ai_story_engine or "").lower()
        if "ollama" in engine_lower:
            sb_data = generate_llm_storyboard(
                story=story,
                total_duration=target_duration,
                shot_duration=target_shot,
                style_prompt=style_prompt,
                llm_url=llm_url,
                llm_model=llm_model,
                api_format="ollama",
            )
        elif "openai" in engine_lower:
            sb_data = generate_llm_storyboard(
                story=story,
                total_duration=target_duration,
                shot_duration=target_shot,
                style_prompt=style_prompt,
                llm_url=llm_url,
                llm_model=llm_model,
                api_format="openai",
            )

        if not sb_data:
            sb_data = generate_local_storyboard(
                story=story,
                total_duration=target_duration,
                shot_duration=target_shot,
                style_prompt=style_prompt,
            )

        scenes = sb_data.scenes
        subjects = sb_data.subjects

        # 2. Canonical Reference Image Generation for each Subject (<Picture 1>, <Picture 2>)
        batch_images, b64_list = generate_subject_reference_images(
            subjects,
            width=width,
            height=height,
            model=model,
            clip=clip,
            vae=vae,
            steps=steps_val,
            cfg=cfg_val,
            seed=seed_val,
            negative_prompt=neg_prompt_str,
            external_images=external_images,
        )

        # 3. Assemble Director Timeline JSON
        timeline_dict = build_director_timeline(
            scenes,
            subjects=subjects,
            full_h3_prompt=sb_data.full_h3_prompt,
            director_mode=director_mode,
            width=width,
            height=height,
            fps=24.0,
            ref_max_size=max(width, height),
        )
        timeline_json = json.dumps(timeline_dict, ensure_ascii=False, indent=2)

        # 4. Format Human-Readable Summary (MiniMax H3 Native Structured Prompt)
        script_summary = sb_data.full_h3_prompt

        # 5. Build MMX_STORYBOARD payload
        total_frames = int(timeline_dict.get("totalFrames") or round(target_duration * 24.0))
        storyboard_payload = {
            "version": 1,
            "story": story,
            "total_duration": target_duration,
            "total_frames": total_frames,
            "frame_rate": 24.0,
            "width": width,
            "height": height,
            "director_mode": director_mode,
            "timeline_data": timeline_json,
            "timeline_json": timeline_json,
            "timeline_dict": timeline_dict,
            "subjects": [s.to_dict() for s in subjects],
            "scenes": [s.to_dict() for s in scenes],
            "images": batch_images,
            "script_summary": script_summary,
            "prompt": sb_data.full_h3_prompt,
        }

        # 6. Push UI update event to connected Director nodes
        try:
            from server import PromptServer

            PromptServer.instance.send_sync(
                "minimax_storyboard_generated",
                {
                    "node_id": str(unique_id) if unique_id else None,
                    "timeline_data": timeline_json,
                    "total_duration": target_duration,
                    "num_scenes": len(scenes),
                    "summary": script_summary,
                },
            )
        except Exception:
            pass

        return (storyboard_payload, timeline_json, batch_images, script_summary)
