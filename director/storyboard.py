# MiniMax H3 AI Story Director — Storyboard and Narrative Video Planner
# Author: Jonathan97480 & Contributors
# Licensed under GNU GPL v3.0

"""AI Storyboard generation and narrative video planning for MiniMax H3 Motion Director.

Transforms high-level narrative stories and scripts into structured, multi-segment
director timelines with per-scene prompts, camera trajectories, durations, continuity
links, and generated reference images.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.storyboard")

MMX_STORYBOARD = "MMX_STORYBOARD"

ASPECT_RATIOS = {
    "16:9 (864x480)": (864, 480),
    "16:9 HD (1280x720)": (1280, 720),
    "9:16 (480x864)": (480, 864),
    "9:16 HD (720x1280)": (720, 1280),
    "1:1 (768x768)": (768, 768),
    "21:9 Cinema (1008x432)": (1008, 432),
}

DEFAULT_STYLE_PROMPT = (
    "Cinematic lighting, photorealistic 8k, Unreal Engine 5 render, "
    "anamorphic lens flare, sharp focus, masterpiece, highly detailed, award-winning cinematography"
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed faces, bad anatomy, text, watermark, amateur, noisy"
)

SHOT_TYPES = [
    ("Plan d'ensemble (Wide Establishing Shot)", "Wide cinematic establishing shot showing the environment, scale, and atmosphere"),
    ("Plan moyen (Medium Shot)", "Medium shot capturing the character interacting with the environment"),
    ("Plan rapproché (Medium Close-Up)", "Medium close-up focusing on character expression, tension, and subtle movements"),
    ("Gros plan dramatique (Close-Up)", "Dramatic close-up on key detail, facial reaction, or decisive object"),
    ("Plan en contre-plongée (Low Angle Shot)", "Low angle heroic perspective emphasizing dominance, stature, and grandeur"),
    ("Plan séquence fluide (Tracking Sequence Shot)", "Dynamic tracking shot following motion smoothly through the setting"),
    ("Plan de révélation (Reveal / Wide Resolution)", "Expansive wide shot revealing unexpected narrative element or peaceful conclusion"),
]

CAMERA_MOTIONS = [
    "Traveling avant fluide vers le sujet (Slow Smooth Dolly In)",
    "Panoramique latéral révélant la scène (Slow Horizontal Pan)",
    "Traveling arrière découvrant l'environnement (Pull-Back Dolly Out)",
    "Mouvement de grue descendant majestueux (Smooth Crane Down)",
    "Caméra portée immersive et subtile (Subtle Handheld Tracking)",
    "Mouvement circulaire orbital autour du sujet (Slow Orbiting Arc Shot)",
    "Traveling latéral suivant la marche (Tracking Sideways Dolly)",
]

LIGHTING_MOODS = [
    "Volumetric neon cyan and magenta illumination with wet rain reflections and atmospheric mist",
    "Warm golden hour sunlight filtering through dust particles with dramatic rim lighting",
    "Deep chiaroscuro shadows with focused spotlight cuts and rich ambient darkness",
    "Soft diffused cinematic natural light with gentle haze and lifelike tonal gradation",
    "Cold eerie moonlight with spectral cyan highlights and drifting smoke",
    "High-contrast anamorphic sunlight with sharp shadows and subtle lens flares",
]


@dataclass
class StoryboardScene:
    index: int
    title: str
    duration: float
    shot_type: str
    camera_motion: str
    lighting: str
    visual_prompt: str
    full_prompt: str
    continuity_visual: bool = True
    continuity_audio: bool = True
    image_tensor: torch.Tensor | None = None
    image_b64: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "duration": self.duration,
            "shot_type": self.shot_type,
            "camera_motion": self.camera_motion,
            "lighting": self.lighting,
            "visual_prompt": self.visual_prompt,
            "full_prompt": self.full_prompt,
            "continuity_visual": self.continuity_visual,
            "continuity_audio": self.continuity_audio,
            "image_b64": self.image_b64,
        }


def parse_aspect_ratio(aspect_str: str) -> tuple[int, int]:
    """Parse aspect ratio combo string into (width, height)."""
    if aspect_str in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_str]
    match = re.search(r"(\d+)\s*[xX*]\s*(\d+)", aspect_str)
    if match:
        w = max(32, int(match.group(1)) // 32 * 32)
        h = max(32, int(match.group(2)) // 32 * 32)
        return w, h
    return 864, 480


def tensor_to_b64_jpeg(tensor: torch.Tensor, quality: int = 88) -> str:
    """Convert (1, H, W, 3) or (H, W, 3) float tensor [0, 1] to base64 JPEG string."""
    if tensor is None:
        return ""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def render_storyboard_card(
    *,
    scene_num: int,
    total_scenes: int,
    title: str,
    duration: float,
    shot_type: str,
    camera_motion: str,
    lighting: str,
    story_snippet: str,
    width: int = 864,
    height: int = 480,
) -> torch.Tensor:
    """Render a stylized cinematic slate card image for a storyboard scene."""
    img = Image.new("RGB", (width, height), color=(14, 18, 26))
    draw = ImageDraw.Draw(img)

    # Subtle gradient background
    for y in range(height):
        ratio = y / max(1, height)
        r = int(14 + ratio * 15)
        g = int(18 + ratio * 20)
        b = int(26 + ratio * 32)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Outer frame border
    border_color = (48, 65, 92)
    accent_color = (0, 180, 216)
    draw.rectangle([(12, 12), (width - 13, height - 13)], outline=border_color, width=2)
    draw.rectangle([(16, 16), (width - 17, height - 17)], outline=(30, 42, 60), width=1)

    # Corner marks (viewfinder style)
    c_len = min(24, width // 20)
    for cx, cy in [(20, 20), (width - 20, 20), (20, height - 20), (width - 20, height - 20)]:
        sx = 1 if cx < width // 2 else -1
        sy = 1 if cy < height // 2 else -1
        draw.line([(cx, cy), (cx + sx * c_len, cy)], fill=accent_color, width=2)
        draw.line([(cx, cy), (cx, cy + sy * c_len)], fill=accent_color, width=2)

    # Header bar
    header_h = 44
    draw.rectangle([(20, 22), (width - 20, 22 + header_h)], fill=(20, 29, 44))
    draw.line([(20, 22 + header_h), (width - 20, 22 + header_h)], fill=accent_color, width=2)

    # Scene title text
    header_text = f"SCÈNE {scene_num}/{total_scenes}  •  {shot_type.split('(')[0].strip()}"
    draw.text((32, 34), header_text, fill=(240, 246, 255))

    # Duration badge
    duration_badge = f"⏱️ {duration:.1f}s"
    draw.text((width - 110, 34), duration_badge, fill=(255, 200, 87))

    # Info boxes
    y_cursor = 82
    box_w = width - 64

    # Camera movement banner
    draw.rectangle([(32, y_cursor), (32 + box_w, y_cursor + 32)], fill=(25, 36, 54), outline=(45, 62, 90))
    cam_label = f"🎥 Caméra : {camera_motion}"
    draw.text((42, y_cursor + 8), cam_label[:90], fill=(120, 220, 255))
    y_cursor += 42

    # Lighting banner
    draw.rectangle([(32, y_cursor), (32 + box_w, y_cursor + 32)], fill=(25, 36, 54), outline=(45, 62, 90))
    light_label = f"💡 Éclairage : {lighting}"
    draw.text((42, y_cursor + 8), light_label[:90], fill=(255, 220, 140))
    y_cursor += 50

    # Story Narrative Snippet Box
    desc_h = max(60, height - y_cursor - 55)
    draw.rectangle([(32, y_cursor), (32 + box_w, y_cursor + desc_h)], fill=(18, 25, 38), outline=(38, 52, 75))
    draw.text((42, y_cursor + 10), "📜 Description du plan :", fill=(170, 190, 215))

    # Wrap story snippet text
    words = story_snippet.split()
    wrapped_lines = []
    current_line = []
    max_line_len = max(35, width // 15)
    for w in words:
        if sum(len(x) + 1 for x in current_line) + len(w) <= max_line_len:
            current_line.append(w)
        else:
            wrapped_lines.append(" ".join(current_line))
            current_line = [w]
    if current_line:
        wrapped_lines.append(" ".join(current_line))

    text_y = y_cursor + 32
    for line in wrapped_lines[:5]:
        draw.text((44, text_y), line, fill=(230, 238, 248))
        text_y += 20

    # Footer
    footer_text = "MiniMax H3 Motion Director  •  Storyboard IA Génératif"
    draw.text((width // 2 - 160, height - 32), footer_text, fill=(80, 105, 135))

    # Convert to Tensor
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def generate_local_scenes(
    story: str,
    num_scenes: int,
    total_duration: float,
    style_prompt: str,
) -> list[StoryboardScene]:
    """Generate intelligent storyboard scenes algorithmically without needing an external LLM server."""
    clean_story = (story or "").strip()
    if not clean_story:
        clean_story = (
            "Un voyageur solitaire traverse une métropole futuriste baignée de pluie néon "
            "et découvre un sanctuaire antique caché au cœur des gratte-ciels."
        )

    # Split story into narrative sentences / beats
    raw_beats = [s.strip() for s in re.split(r"[.!?;\n]+", clean_story) if s.strip()]
    if not raw_beats:
        raw_beats = [clean_story]

    # Sub-split long beats if fewer beats than requested scenes
    if len(raw_beats) < num_scenes:
        expanded_beats = []
        for beat in raw_beats:
            sub = [c.strip() for c in re.split(r",\s*(?:alors|puis|ensuite|soudain|mais|et|pendant que|while|then|suddenly|and)\s+", beat, flags=re.IGNORECASE) if c.strip()]
            if len(sub) > 1:
                expanded_beats.extend(sub)
            else:
                expanded_beats.append(beat)
        raw_beats = expanded_beats if expanded_beats else raw_beats

    # Distribute beats across num_scenes
    per_scene_beats: list[str] = []
    for i in range(num_scenes):
        if i < len(raw_beats):
            per_scene_beats.append(raw_beats[i])
        else:
            # Re-use with thematic progression
            base = raw_beats[i % len(raw_beats)]
            per_scene_beats.append(f"{base} (Suite et développement de l'action)")

    shot_duration = round(total_duration / max(1, num_scenes), 2)
    scenes: list[StoryboardScene] = []

    # Detect story atmosphere
    lower_story = clean_story.lower()
    if any(k in lower_story for k in ["neon", "cyberpunk", "tokyo", "futur", "robot", "pluie", "rain"]):
        base_lighting = LIGHTING_MOODS[0]
    elif any(k in lower_story for k in ["soleil", "sunset", "or", "golden", "nature", "forêt", "prairie"]):
        base_lighting = LIGHTING_MOODS[1]
    elif any(k in lower_story for k in ["nuit", "sombre", "ombre", "peur", "dark", "mystère", "temple"]):
        base_lighting = LIGHTING_MOODS[2]
    else:
        base_lighting = LIGHTING_MOODS[3]

    for i in range(num_scenes):
        idx = i + 1
        beat_text = per_scene_beats[i]
        shot_type, shot_desc = SHOT_TYPES[min(i, len(SHOT_TYPES) - 1)]
        camera = CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)]
        lighting = LIGHTING_MOODS[(i + 1) % len(LIGHTING_MOODS)] if i > 0 else base_lighting

        # Build optimized prompt for MiniMax H3
        cam_short = camera.split("(")[-1].rstrip(")") if "(" in camera else camera
        clean_beat = beat_text.rstrip(".")

        full_prompt = (
            f"[Camera: {cam_short}] Cinematic shot of {clean_beat}. "
            f"{shot_desc}. Lighting: {lighting}. Style: {style_prompt}."
        )

        scenes.append(
            StoryboardScene(
                index=idx,
                title=f"Plan {idx} : {shot_type.split('(')[0].strip()}",
                duration=shot_duration,
                shot_type=shot_type,
                camera_motion=camera,
                lighting=lighting,
                visual_prompt=clean_beat,
                full_prompt=full_prompt,
                continuity_visual=(i > 0),
                continuity_audio=(i > 0),
            )
        )

    return scenes


def generate_llm_scenes(
    *,
    story: str,
    num_scenes: int,
    total_duration: float,
    style_prompt: str,
    llm_url: str = "http://127.0.0.1:11434",
    llm_model: str = "qwen2.5:7b",
    api_format: str = "ollama",
    timeout: int = 40,
) -> list[StoryboardScene] | None:
    """Generate storyboard scenes using a local LLM (Ollama or OpenAI-compatible)."""
    shot_duration = round(total_duration / max(1, num_scenes), 2)
    system_prompt = (
        "You are an award-winning cinematic director and AI video storyboard artist. "
        "Your task is to break down the user's narrative story into a structured sequence "
        f"of exactly {num_scenes} sequential video shots for MiniMax H3.\n"
        "Return ONLY a valid JSON object with the following schema, and no extra markdown or chit-chat:\n"
        "{\n"
        '  "title": "Short title",\n'
        '  "scenes": [\n'
        "    {\n"
        '      "scene_index": 1,\n'
        '      "shot_type": "Wide Establishing Shot",\n'
        '      "camera_motion": "Slow Dolly In",\n'
        '      "lighting": "Volumetric neon cyan with wet reflections",\n'
        '      "visual_prompt": "Cinematic visual description of subject, action and setting",\n'
        '      "continuity_visual": true,\n'
        '      "continuity_audio": true\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_message = (
        f"Story to adapt:\n{story}\n\n"
        f"Parameters:\n"
        f"- Target number of scenes: {num_scenes}\n"
        f"- Duration per scene: {shot_duration} seconds\n"
        f"- Visual aesthetic style: {style_prompt}\n"
        "Write visually evocative, precise prompts focusing on movement and cinematography."
    )

    clean_url = llm_url.rstrip("/")
    payload: dict[str, Any] = {}
    endpoint = ""

    if "ollama" in api_format.lower():
        endpoint = f"{clean_url}/api/chat"
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 8192},
        }
    else:
        endpoint = f"{clean_url}/chat/completions" if clean_url.endswith("/v1") else f"{clean_url}/v1/chat/completions"
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "ComfyUI-MiniMax-H3-StoryDirector"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_json = json.loads(response.read().decode("utf-8"))

        content_str = ""
        if "ollama" in api_format.lower():
            content_str = res_json.get("message", {}).get("content", "")
        else:
            choices = res_json.get("choices") or []
            if choices:
                content_str = choices[0].get("message", {}).get("content", "")

        # Strip markdown fences if present
        clean_content = re.sub(r"^```(?:json)?\s*", "", content_str.strip())
        clean_content = re.sub(r"\s*```$", "", clean_content).strip()

        parsed = json.loads(clean_content)
        raw_scenes = parsed.get("scenes") or []
        if not raw_scenes:
            return None

        scenes: list[StoryboardScene] = []
        for i, raw in enumerate(raw_scenes[:num_scenes]):
            idx = i + 1
            stype = str(raw.get("shot_type") or SHOT_TYPES[min(i, len(SHOT_TYPES) - 1)][0])
            cam = str(raw.get("camera_motion") or CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)])
            lighting = str(raw.get("lighting") or LIGHTING_MOODS[i % len(LIGHTING_MOODS)])
            vprompt = str(raw.get("visual_prompt") or f"Scene {idx} action").strip()

            full_prompt = (
                f"[Camera: {cam}] {vprompt}. Shot: {stype}. Lighting: {lighting}. Style: {style_prompt}."
            )

            scenes.append(
                StoryboardScene(
                    index=idx,
                    title=f"Plan {idx} : {stype}",
                    duration=shot_duration,
                    shot_type=stype,
                    camera_motion=cam,
                    lighting=lighting,
                    visual_prompt=vprompt,
                    full_prompt=full_prompt,
                    continuity_visual=bool(raw.get("continuity_visual", i > 0)),
                    continuity_audio=bool(raw.get("continuity_audio", i > 0)),
                )
            )

        # Pad if LLM returned fewer scenes than requested
        while len(scenes) < num_scenes:
            pad_i = len(scenes)
            scenes.append(
                StoryboardScene(
                    index=pad_i + 1,
                    title=f"Plan {pad_i + 1}",
                    duration=shot_duration,
                    shot_type=SHOT_TYPES[min(pad_i, len(SHOT_TYPES) - 1)][0],
                    camera_motion=CAMERA_MOTIONS[pad_i % len(CAMERA_MOTIONS)],
                    lighting=LIGHTING_MOODS[pad_i % len(LIGHTING_MOODS)],
                    visual_prompt=f"Continuation of sequence shot {pad_i + 1}",
                    full_prompt=f"Continuation of sequence shot {pad_i + 1}. Style: {style_prompt}.",
                    continuity_visual=True,
                    continuity_audio=True,
                )
            )

        return scenes

    except Exception as exc:
        log.warning("LLM Storyboard call failed (%s); falling back to Built-in Engine: %s", endpoint, exc)
        return None


def generate_scene_reference_images(
    scenes: list[StoryboardScene],
    *,
    width: int,
    height: int,
    model=None,
    clip=None,
    vae=None,
    steps: int = 20,
    cfg: float = 6.5,
    seed: int = 0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    external_images: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Generate image references for each scene using diffusion model, external batch, or storyboard cards."""
    generated_tensors: list[torch.Tensor] = []
    b64_list: list[str] = []

    has_diffusion = model is not None and clip is not None and vae is not None
    ext_count = int(external_images.shape[0]) if external_images is not None and external_images.ndim == 4 else 0

    for i, scene in enumerate(scenes):
        img_tensor: torch.Tensor | None = None

        # Option A: Connect to ComfyUI diffusion pipeline
        if has_diffusion:
            try:
                from nodes import common_ksampler

                # Encode conditioning
                pos_tokens = clip.tokenize(scene.full_prompt)
                pos_cond = clip.encode_from_tokens(pos_tokens, return_pooled=True)
                if not isinstance(pos_cond, (list, tuple)):
                    pos_cond = [pos_cond]
                pos_cond = [[pos_cond[0], {}]] if isinstance(pos_cond[0], torch.Tensor) else pos_cond

                neg_tokens = clip.tokenize(negative_prompt)
                neg_cond = clip.encode_from_tokens(neg_tokens, return_pooled=True)
                if not isinstance(neg_cond, (list, tuple)):
                    neg_cond = [neg_cond]
                neg_cond = [[neg_cond[0], {}]] if isinstance(neg_cond[0], torch.Tensor) else neg_cond

                # Detect latent channels (16 for Flux/SD3, 4 for SDXL/SD1.5)
                latent_channels = 4
                if hasattr(model, "model") and hasattr(model.model, "latent_format"):
                    latent_channels = getattr(model.model.latent_format, "latent_channels", 4) or 4

                latent_w = max(4, width // 8)
                latent_h = max(4, height // 8)
                empty_latent = {"samples": torch.zeros([1, latent_channels, latent_h, latent_w])}

                sampled = common_ksampler(
                    model=model,
                    seed=seed + i,
                    steps=steps,
                    cfg=cfg,
                    sampler_name="euler",
                    scheduler="normal",
                    positive=pos_cond,
                    negative=neg_cond,
                    latent=empty_latent,
                    denoise=1.0,
                )
                decoded = vae.decode(sampled[0]["samples"])
                img_tensor = decoded[:1]
            except Exception as diff_exc:
                log.warning("Diffusion reference image generation failed for Scene %d: %s. Using storyboard card.", i + 1, diff_exc)
                img_tensor = None

        # Option B: Use external image batch if supplied
        if img_tensor is None and ext_count > 0:
            pick_idx = min(i, ext_count - 1)
            img_tensor = external_images[pick_idx : pick_idx + 1]

        # Option C: Fallback to stylized high-contrast storyboard card
        if img_tensor is None:
            img_tensor = render_storyboard_card(
                scene_num=scene.index,
                total_scenes=len(scenes),
                title=scene.title,
                duration=scene.duration,
                shot_type=scene.shot_type,
                camera_motion=scene.camera_motion,
                lighting=scene.lighting,
                story_snippet=scene.visual_prompt,
                width=width,
                height=height,
            )

        scene.image_tensor = img_tensor
        b64_str = tensor_to_b64_jpeg(img_tensor)
        scene.image_b64 = b64_str
        generated_tensors.append(img_tensor)
        b64_list.append(b64_str)

    batch_tensor = torch.cat(generated_tensors, dim=0) if generated_tensors else torch.zeros([1, height, width, 3])
    return batch_tensor, b64_list


def build_director_timeline(
    scenes: list[StoryboardScene],
    *,
    director_mode: str = "mixed",
    width: int = 864,
    height: int = 480,
    fps: float = 24.0,
    ref_max_size: int = 864,
) -> dict[str, Any]:
    """Assemble a compliant MiniMax H3 Motion Director timeline specification."""
    total_frames = sum(max(5, int(round(s.duration * fps))) for s in scenes)

    if director_mode.lower() == "mixed":
        mixed_segments = []
        for s in scenes:
            seg_dict: dict[str, Any] = {
                "id": f"seg_{s.index}",
                "mode": "i2v",
                "prompt": s.full_prompt,
                "duration": float(s.duration),
                "inputs": {
                    "resultRefs": [],
                    "startFrame": {
                        "imageB64": s.image_b64,
                    },
                    "identityPictures": [],
                },
                "continuity": {
                    "visual": bool(s.continuity_visual),
                    "audio": bool(s.continuity_audio),
                },
            }
            mixed_segments.append(seg_dict)

        timeline_dict = {
            "version": 1,
            "timelineMode": "mixed",
            "frameRate": float(fps),
            "width": int(width),
            "height": int(height),
            "refMaxSize": int(ref_max_size),
            "output": {
                "mode": "fixed",
                "longEdge": int(ref_max_size),
                "width": int(width),
                "height": int(height),
                "maxExportFrames": 0,
                "exportMode": "all",
                "audioMode": "generate",
            },
            "runSelectEnabled": False,
            "runSelection": [],
            "segments": mixed_segments,
        }
        return timeline_dict

    # Standard / Global timeline structure
    cursor = 0
    std_segments = []
    for s in scenes:
        seg_frames = max(5, int(round(s.duration * fps)))
        std_segments.append(
            {
                "id": f"s{s.index - 1}",
                "start": cursor,
                "length": seg_frames,
                "prompt": s.full_prompt,
                "taskType": director_mode,
                "refs": [{"index": 1, "imageB64": s.image_b64}] if s.image_b64 else [],
                "referenceVideo": {},
            }
        )
        cursor += seg_frames

    return {
        "version": 4,
        "editMode": "global",
        "totalFrames": total_frames,
        "frameRate": float(fps),
        "width": int(width),
        "height": int(height),
        "refMaxSize": int(ref_max_size),
        "output": {
            "mode": "fixed",
            "longEdge": int(ref_max_size),
            "width": int(width),
            "height": int(height),
            "maxExportFrames": 0,
            "exportMode": "all",
            "audioMode": "generate",
        },
        "videoClips": [],
        "video": {"fileName": "", "videoFile": "", "subfolder": "", "type": "input", "frames": [], "frameMap": []},
        "global": {"taskType": director_mode, "prompt": scenes[0].full_prompt if scenes else "", "refs": []},
        "segments": std_segments,
    }


def format_script_summary(scenes: list[StoryboardScene], total_duration: float, title: str = "") -> str:
    """Format a clean human-readable script breakdown."""
    lines = [
        f"═══════════════════════════════════════════════════════════════",
        f"🎬 SCÉNARIO & STORYBOARD IA - MINIMAX H3 MOTION DIRECTOR",
        f"═══════════════════════════════════════════════════════════════",
        f"Durée totale : {total_duration:.1f} secondes  |  Nombre de plans : {len(scenes)}",
        f"---------------------------------------------------------------",
    ]
    for s in scenes:
        lines.append(f"▶ PLAN {s.index} ({s.duration:.1f}s) — {s.shot_type}")
        lines.append(f"  • Caméra   : {s.camera_motion}")
        lines.append(f"  • Éclairage: {s.lighting}")
        lines.append(f"  • Prompt   : {s.full_prompt}")
        lines.append(f"  • Continuité: Visuelle = {'Oui' if s.continuity_visual else 'Non'}, Audio = {'Oui' if s.continuity_audio else 'Non'}")
        lines.append("")
    return "\n".join(lines)
