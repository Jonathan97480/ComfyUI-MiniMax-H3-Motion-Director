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
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
class StoryboardSubject:
    index: int
    name: str
    category: str  # "character", "environment", "prop"
    definition: str
    retention: str
    image_prompt: str
    image_tensor: torch.Tensor | None = None
    image_b64: str = ""
    image_file: str = ""

    @property
    def subject_tag(self) -> str:
        return f"<Subject {self.index}>"

    @property
    def picture_tag(self) -> str:
        return f"<Picture {self.index}>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "category": self.category,
            "subject_tag": self.subject_tag,
            "picture_tag": self.picture_tag,
            "definition": self.definition,
            "retention": self.retention,
            "image_prompt": self.image_prompt,
            "image_b64": self.image_b64,
            "image_file": self.image_file,
        }


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
    start_time: float = 0.0
    timestamp_str: str = "00:00.000"
    image_prompt: str = ""
    continuity_visual: bool = True
    continuity_audio: bool = True
    image_tensor: torch.Tensor | None = None
    image_b64: str = ""
    image_file: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "start_time": self.start_time,
            "duration": self.duration,
            "timestamp_str": self.timestamp_str,
            "shot_type": self.shot_type,
            "camera_motion": self.camera_motion,
            "lighting": self.lighting,
            "visual_prompt": self.visual_prompt,
            "full_prompt": self.full_prompt,
            "image_prompt": self.image_prompt,
            "continuity_visual": self.continuity_visual,
            "continuity_audio": self.continuity_audio,
            "image_b64": self.image_b64,
            "image_file": self.image_file,
        }


@dataclass
class StoryboardData:
    subjects: list[StoryboardSubject]
    scenes: list[StoryboardScene]
    summary: str
    detailed_description_intro: str
    soundscape: str
    music: str = "N/A"
    full_h3_prompt: str = ""


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


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.mmm string (e.g. 00:03.000)."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{m:02d}:{s:02d}.{ms:03d}"


def parse_timestamp_str(ts_str: str) -> float:
    """Parse MM:SS.mmm or SS.mmm into float seconds."""
    ts_str = ts_str.strip()
    try:
        if ":" in ts_str:
            parts = ts_str.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(ts_str)
    except Exception:
        return 0.0


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


def get_comfy_core():
    """Safely return ComfyUI's root nodes module without shadowing by custom_nodes subpackages."""
    import sys
    if "nodes" in sys.modules and hasattr(sys.modules["nodes"], "common_ksampler"):
        return sys.modules["nodes"]
    try:
        import folder_paths
        import importlib.util
        comfy_root = os.path.dirname(folder_paths.__file__)
        nodes_py = os.path.join(comfy_root, "nodes.py")
        if os.path.isfile(nodes_py):
            spec = importlib.util.spec_from_file_location("comfy_core_nodes", nodes_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        pass
    return None


def clean_subject_def_for_image(text: str) -> str:
    """Remove video prompt constraints to produce clean text-to-image prompts."""
    clean = re.sub(r"Preserve\s+.+?(?:\.|$)", "", text, flags=re.IGNORECASE)
    clean = re.sub(r"The background of <Picture\s*\d+>\s+is not used\.?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"The character shown in <Picture\s*\d+>.*?controls.*?identity\.?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"The layout is fixed:?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"the exact\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"shown in the supplied reference,?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip(". ")
    return clean


def render_storyboard_visual_concept(
    *,
    scene_num: int = 1,
    total_scenes: int = 2,
    title: str = "Subject",
    story_snippet: str = "",
    lighting: str = "",
    category: str = "environment",
    width: int = 864,
    height: int = 480,
    seed: int = 0,
) -> torch.Tensor:
    """Render a stylized visual concept art illustration (character turnaround sheet or environment scenery)."""
    rng = random.Random(seed + scene_num * 101)
    img = Image.new("RGB", (width, height), color=(12, 16, 24))
    draw = ImageDraw.Draw(img)

    lower_snippet = (story_snippet + " " + lighting + " " + title).lower()
    is_cyber = any(k in lower_snippet for k in ["neon", "cyber", "robot", "pluie", "rain", "tokyo", "futur", "plasma"])
    is_sunset = any(k in lower_snippet for k in ["soleil", "sunset", "golden", "or", "nature", "forêt", "wood"])
    accent_col = (0, 210, 255) if is_cyber else ((255, 175, 60) if is_sunset else (150, 190, 255))

    if category == "character":
        # Multi-angle Character Turnaround Model Sheet (Front, 3/4, Profile) on clean studio background
        panel_w = width // 3
        for y in range(height - 30):
            ratio = y / max(1, height)
            r = int(12 + ratio * 16)
            g = int(16 + ratio * 18)
            b = int(24 + ratio * 28)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        for p in range(1, 3):
            x = p * panel_w
            draw.line([(x, 35), (x, height - 35)], fill=(38, 50, 72), width=2)

        views = ["FRONT VIEW", "THREE-QUARTER VIEW", "PROFILE VIEW"]
        for idx, view_name in enumerate(views):
            center_x = idx * panel_w + panel_w // 2
            cy = int((height - 35) * 0.52)
            head_r = int(min(panel_w, height) * 0.11)
            head_y = cy - head_r * 2

            halo_rad = int(head_r * 2.2)
            for r_step in range(halo_rad, 0, -6):
                af = (1.0 - r_step / halo_rad) ** 1.8
                draw.ellipse(
                    [(center_x - r_step, cy - r_step), (center_x + r_step, cy + r_step)],
                    fill=(int(accent_col[0] * af * 0.35), int(accent_col[1] * af * 0.35), int(accent_col[2] * af * 0.35)),
                )

            if idx == 0:
                draw.ellipse([(center_x - head_r, head_y - head_r), (center_x + head_r, head_y + head_r)], fill=(20, 26, 40), outline=accent_col, width=2)
                draw.polygon([
                    (center_x - int(head_r * 2.0), height - 42),
                    (center_x - int(head_r * 1.3), head_y + head_r),
                    (center_x + int(head_r * 1.3), head_y + head_r),
                    (center_x + int(head_r * 2.0), height - 42)
                ], fill=(20, 26, 40), outline=(50, 70, 100), width=1)
            elif idx == 1:
                draw.ellipse([(center_x - head_r, head_y - head_r), (center_x + head_r, head_y + head_r)], fill=(20, 26, 40), outline=accent_col, width=2)
                draw.polygon([
                    (center_x - int(head_r * 2.2), height - 42),
                    (center_x - int(head_r * 1.5), head_y + head_r),
                    (center_x + int(head_r * 0.9), head_y + head_r),
                    (center_x + int(head_r * 1.6), height - 42)
                ], fill=(20, 26, 40), outline=(50, 70, 100), width=1)
            else:
                draw.ellipse([(center_x - int(head_r * 0.8), head_y - head_r), (center_x + int(head_r * 0.8), head_y + head_r)], fill=(20, 26, 40), outline=accent_col, width=2)
                draw.polygon([
                    (center_x - int(head_r * 1.4), height - 42),
                    (center_x - int(head_r * 0.9), head_y + head_r),
                    (center_x + int(head_r * 0.7), head_y + head_r),
                    (center_x + int(head_r * 1.2), height - 42)
                ], fill=(20, 26, 40), outline=(50, 70, 100), width=1)

            draw.text((center_x - 45, 14), view_name, fill=(160, 190, 230))

        draw.rectangle([(0, height - 28), (width, height)], fill=(8, 12, 18))
        draw.text((16, height - 22), f"<Subject {scene_num}> CHARACTER MODEL SHEET: {title.upper()} (TURNAROUND MULTI-ANGLE REFERENCE)", fill=(0, 220, 250))
    else:
        # Environment architectural layout illustration (pure empty environment, NO people)
        horizon_y = int(height * 0.62)
        for y in range(horizon_y):
            ratio = y / max(1, horizon_y)
            if is_cyber:
                r, g, b = int(12 + ratio * 45), int(16 + ratio * 20), int(45 + ratio * 85)
            elif is_sunset:
                r, g, b = int(50 + ratio * 150), int(25 + ratio * 80), int(20 + ratio * 40)
            else:
                r, g, b = int(15 + ratio * 35), int(22 + ratio * 45), int(40 + ratio * 75)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        desk_x2 = int(width * 0.42)
        draw.rectangle([(int(width * 0.08), int(height * 0.22)), (desk_x2, horizon_y)], fill=(16, 20, 30), outline=(35, 55, 80))
        bed_x1 = int(width * 0.58)
        draw.rectangle([(bed_x1, int(height * 0.35)), (int(width * 0.92), horizon_y)], fill=(20, 24, 34), outline=(35, 55, 80))

        for y in range(horizon_y, height - 28):
            ratio = (y - horizon_y) / max(1, height - horizon_y)
            draw.line([(0, y), (width, y)], fill=(int(14 + ratio * 18), int(16 + ratio * 18), int(28 + ratio * 32)))

        draw.rectangle([(16, 16), (width - 17, height - 32)], outline=(35, 55, 80), width=1)
        draw.rectangle([(0, height - 28), (width, height)], fill=(8, 12, 18))
        draw.text((16, height - 22), f"<Subject {scene_num}> ENVIRONMENT ARCHITECTURAL REFERENCE: {title.upper()} (EMPTY LAYOUT)", fill=(0, 220, 250))

    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
def format_h3_structured_prompt(
    subjects: list[StoryboardSubject],
    summary: str,
    detailed_description_intro: str,
    scenes: list[StoryboardScene],
    soundscape: str,
    music: str = "N/A",
) -> str:
    """Format prompt according to MiniMax H3's official multi-subject cinematic reference schema."""
    lines: list[str] = []
    lines.append("subject_definitions:\n")
    for subj in subjects:
        lines.append(f"{subj.subject_tag} {subj.name}, based on {subj.picture_tag}: {subj.definition}\n")

    lines.append("summary:\n")
    clean_summary = summary.strip()
    if not clean_summary.startswith("[reference generation]"):
        clean_summary = f"[reference generation] {clean_summary}"
    lines.append(f"{clean_summary}\n")

    lines.append("retention_analysis:\n")
    for subj in subjects:
        lines.append(f"{subj.subject_tag} (appears throughout): {subj.retention}\n")

    lines.append("detailed_description:\n")
    if detailed_description_intro:
        lines.append(f"{detailed_description_intro.strip()}\n")

    for i, s in enumerate(scenes):
        if i == 0:
            lines.append(f"[Shot 1] {s.visual_prompt.strip()}\n")
        else:
            ts_str = format_timestamp(s.start_time)
            cut_text = s.visual_prompt.strip()
            if not cut_text.lower().startswith("at approximately") and not cut_text.lower().startswith("make a"):
                cut_text = f"make a hard cinematic cut to {cut_text}"
            lines.append(f"At approximately {ts_str}, {cut_text}\n")

    if soundscape:
        lines.append("overall_soundscape:\n")
        lines.append(f"{soundscape.strip()}\n")

    lines.append("non_diegetic_music:\n")
    lines.append(f"{music or 'N/A'}")

    return "\n".join(lines)


def parse_existing_h3_prompt(text: str, total_duration: float = 15.0, style_prompt: str = "") -> StoryboardData | None:
    """Parse an already structured MiniMax H3 multi-subject prompt into StoryboardData."""
    if "subject_definitions:" not in text.lower():
        return None

    subj_pattern = re.compile(
        r"<Subject\s+(\d+)>\s*([^,:\n]+)[,:]\s*based on\s*<Picture\s+(\d+)>:\s*(.+?)(?=(?:<Subject\s+\d+>|summary:|$))",
        re.DOTALL | re.IGNORECASE,
    )
    subjects: list[StoryboardSubject] = []
    for m in subj_pattern.finditer(text):
        idx = int(m.group(1))
        name = m.group(2).strip()
        defn = m.group(4).strip()
        cat = "character" if idx == 1 or any(k in name.lower() for k in ["girl", "boy", "man", "woman", "knight", "fille", "homme", "femme"]) else "environment"
        clean = clean_subject_def_for_image(defn)
        if cat == "character":
            img_p = f"Character design concept art reference sheet portrait and full body view of {name}: {clean}, isolated, clean background, neutral lighting, masterpiece, sharp focus, {style_prompt}"
            retention = f"fully_preserved — preserve the exact facial identity, hairstyle, age appearance, body proportions, clothing and recognizable character design from <Picture {idx}>. Allow natural changes in pose, expression and movement."
        else:
            img_p = f"Architectural environmental concept art establishing view of {name}: {clean}, wide angle view, empty room without people, spatial layout, furniture arrangement, cinematic lighting, masterpiece, sharp focus, {style_prompt}"
            retention = f"fully_preserved — preserve the exact {name.lower()} architecture, spatial layout, materials, furniture arrangement, lighting direction and spatial relationships from <Picture {idx}>."

        subjects.append(
            StoryboardSubject(
                index=idx,
                name=name,
                category=cat,
                definition=defn,
                retention=retention,
                image_prompt=img_p,
            )
        )

    # Extract summary
    summary_m = re.search(r"summary:\s*(.+?)(?=retention_analysis:|$)", text, re.DOTALL | re.IGNORECASE)
    summary = summary_m.group(1).strip() if summary_m else "[reference generation] Cinematic sequence."

    # Extract soundscape & music
    sound_m = re.search(r"overall_soundscape:\s*(.+?)(?=non_diegetic_music:|$)", text, re.DOTALL | re.IGNORECASE)
    soundscape = sound_m.group(1).strip() if sound_m else "Atmospheric soundscape."
    music_m = re.search(r"non_diegetic_music:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    music = music_m.group(1).strip() if music_m else "N/A"

    # Extract detailed description intro and shots
    desc_m = re.search(r"detailed_description:\s*(.+?)(?=overall_soundscape:|$)", text, re.DOTALL | re.IGNORECASE)
    desc_body = desc_m.group(1).strip() if desc_m else ""

    shot_pattern = re.compile(
        r"(?:\[Shot\s+(\d+)\]|At approximately\s+(\d{1,2}:\d{2}(?:\.\d+)?)[,\s]+(?:make a (?:hard\s+)?(?:cinematic\s+)?cut to|continue in))\s*(.+?)(?=(?:\[Shot|\bAt approximately|\bEnd with|$))",
        re.DOTALL | re.IGNORECASE,
    )
    raw_shots = list(shot_pattern.finditer(desc_body))
    scenes: list[StoryboardScene] = []

    intro_end = raw_shots[0].start() if raw_shots else 0
    intro_text = desc_body[:intro_end].strip()

    if raw_shots:
        for i, m in enumerate(raw_shots):
            idx = i + 1
            ts_str = m.group(2) or "00:00.000"
            start_t = parse_timestamp_str(ts_str)
            content = m.group(3).strip()
            next_t = parse_timestamp_str(raw_shots[i + 1].group(2) or "00:00.000") if i + 1 < len(raw_shots) else total_duration
            duration = max(1.5, next_t - start_t)

            scenes.append(
                StoryboardScene(
                    index=idx,
                    title=f"Plan {idx}",
                    start_time=start_t,
                    duration=duration,
                    timestamp_str=ts_str,
                    shot_type="Cinematic Cut",
                    camera_motion="Cinematic Camera Motion",
                    lighting="Atmospheric",
                    visual_prompt=content,
                    full_prompt=content,
                    image_prompt=content,
                    continuity_visual=(i > 0),
                    continuity_audio=(i > 0),
                )
            )
    else:
        scenes.append(
            StoryboardScene(
                index=1,
                title="Plan 1",
                start_time=0.0,
                duration=total_duration,
                timestamp_str="00:00.000",
                shot_type="Plan d'ensemble",
                camera_motion="Static",
                lighting="Cinematic",
                visual_prompt=desc_body or summary,
                full_prompt=desc_body or summary,
                image_prompt=desc_body or summary,
                continuity_visual=False,
                continuity_audio=False,
            )
        )

    return StoryboardData(
        subjects=subjects,
        scenes=scenes,
        summary=summary,
        detailed_description_intro=intro_text,
        soundscape=soundscape,
        music=music,
        full_h3_prompt=text.strip(),
    )


def calculate_cinematic_shot_durations(
    total_duration: float,
    max_shot_duration: float,
    num_beats: int = 4,
) -> list[float]:
    """Calculate dynamically varying shot durations honoring max_shot_duration as a strict upper bound.

    Shot durations vary according to dramatic pacing (establishing shots and climax/resolutions held longer,
    reaction/detail shots punchier and shorter), while guaranteeing:
    - Each shot duration <= max_shot_duration
    - Minimum shot duration >= 2.0s
    - Sum of all shot durations == total_duration exactly
    """
    total = max(2.0, float(total_duration))
    max_dur = max(2.0, float(max_shot_duration))

    min_shots = max(1, math.ceil(total / max_dur))
    max_possible_shots = max(1, int(total // 2.5))
    target_shots = max(min_shots, min(max_possible_shots, max(num_beats, min_shots)))

    if target_shots <= 1:
        return [round(total, 2)]

    base_patterns = {
        2: [1.1, 0.9],
        3: [1.15, 0.85, 1.0],
        4: [1.15, 0.85, 1.05, 0.95],
        5: [1.15, 0.85, 1.05, 0.9, 1.05],
        6: [1.15, 0.85, 1.0, 0.9, 1.05, 1.05],
    }
    pattern = base_patterns.get(target_shots)
    if not pattern:
        pattern = [1.0] * target_shots
        pattern[0] = 1.15
        pattern[1] = 0.85
        pattern[-1] = 1.1

    weight_sum = sum(pattern)
    raw_durations = [(w / weight_sum) * total for w in pattern]

    min_dur = 2.0
    for _ in range(10):
        clamped = False
        excess = 0.0
        free_indices = []
        for idx, d in enumerate(raw_durations):
            if d > max_dur:
                excess += (d - max_dur)
                raw_durations[idx] = max_dur
                clamped = True
            elif d < min_dur and total >= target_shots * min_dur:
                deficit = min_dur - d
                excess -= deficit
                raw_durations[idx] = min_dur
                clamped = True
            else:
                free_indices.append(idx)

        if not clamped or abs(excess) < 1e-4 or not free_indices:
            break

        share = excess / len(free_indices)
        for idx in free_indices:
            raw_durations[idx] += share

    rounded = [round(d, 1) for d in raw_durations]
    diff = round(total - sum(rounded), 2)
    for idx in reversed(range(len(rounded))):
        if rounded[idx] + diff <= max_dur and rounded[idx] + diff >= min_dur:
            rounded[idx] = round(rounded[idx] + diff, 1)
            diff = 0.0
            break
    if abs(diff) > 1e-4:
        rounded[-1] = round(rounded[-1] + diff, 2)

    return rounded


def extract_multi_subjects_from_story(story: str) -> dict[str, Any]:
    """Analyze high-level narrative to extract all characters, companions, environments, and props in order of appearance."""
    lower = story.lower()

    char_candidates = [
        ("chevalier cyberpunk", "Cyberpunk Knight", "armored cyberpunk knight in high-tech mechanical plate armor with glowing LED accents, cybernetic visor and utility cloak", "he"),
        ("cyberpunk knight", "Cyberpunk Knight", "armored cyberpunk knight in high-tech mechanical plate armor with glowing LED accents, cybernetic visor and utility cloak", "he"),
        ("samouraï", "Samurai Warrior", "ronin samurai warrior in traditional lacquered armor plates with patterned haori and katana sheath", "he"),
        ("samourai", "Samurai Warrior", "ronin samurai warrior in traditional lacquered armor plates with patterned haori and katana sheath", "he"),
        ("samurai", "Samurai Warrior", "ronin samurai warrior in traditional lacquered armor plates with patterned haori and katana sheath", "he"),
        ("fille", "Anime Girl", "teenage anime girl with expressive facial identity, distinctive styled hair, casual or school clothing and realistic proportions", "she"),
        ("girl", "Anime Girl", "teenage anime girl with expressive facial identity, distinctive styled hair, casual or school clothing and realistic proportions", "she"),
        ("femme", "Elegant Woman", "young adult woman with elegant features, distinctive hairstyle, tailored modern clothing and natural poise", "she"),
        ("woman", "Elegant Woman", "young adult woman with elegant features, distinctive hairstyle, tailored modern clothing and natural poise", "she"),
        ("homme", "Man", "young adult man with rugged facial features, styled hair, atmospheric jacket and layered garments", "he"),
        ("man", "Man", "young adult man with rugged facial features, styled hair, atmospheric jacket and layered garments", "he"),
        ("robot", "Cybernetic Android", "humanoid android robot with polished metallic chassis, optic sensors, exposed joint hydraulics and sleek design", "it"),
        ("androïde", "Cybernetic Android", "humanoid android robot with polished metallic chassis, optic sensors, exposed joint hydraulics and sleek design", "it"),
        ("android", "Cybernetic Android", "humanoid android robot with polished metallic chassis, optic sensors, exposed joint hydraulics and sleek design", "it"),
        ("mecha", "Mecha Pilot", "tactical mecha pilot in fitted flight suit with biometric neural interface headset", "he"),
        ("mécha", "Mecha Pilot", "tactical mecha pilot in fitted flight suit with biometric neural interface headset", "he"),
        ("ninja", "Tactical Shinobi", "covert shinobi ninja in dark tactical gear, cloth mask, strapped gauntlets and stealth attire", "he"),
        ("guerrier", "Battle Warrior", "battle-hardened warrior with combat armor, survival harness and determined expression", "he"),
        ("warrior", "Battle Warrior", "battle-hardened warrior with combat armor, survival harness and determined expression", "he"),
        ("detective", "Noir Detective", "investigative detective in trench coat, fedora hat, loosened tie and observant eyes", "he"),
        ("détective", "Noir Detective", "investigative detective in trench coat, fedora hat, loosened tie and observant eyes", "he"),
        ("sorcier", "Arcane Mage", "arcane wizard in embroidered mystical robes with runic leather wraps and glowing amulet", "he"),
        ("magicien", "Arcane Mage", "arcane wizard in embroidered mystical robes with runic leather wraps and glowing amulet", "he"),
        ("wizard", "Arcane Mage", "arcane wizard in embroidered mystical robes with runic leather wraps and glowing amulet", "he"),
        ("astronaute", "Astronaut", "explorer astronaut in pressurized EVA space suit with golden visor reflections", "he"),
        ("astronaut", "Astronaut", "explorer astronaut in pressurized EVA space suit with golden visor reflections", "he"),
    ]

    primary_char = None
    for kw, name, desc, pronoun in char_candidates:
        if kw in lower:
            primary_char = (name, desc, pronoun)
            break
    if not primary_char:
        primary_char = ("Protagonist", "cinematic protagonist with striking appearance, detailed tailored clothing and natural posture", "they")

    companion_candidates = [
        ("renard holographique", "Holographic Fox", "luminescent ethereal holographic fox companion with neon data particles and glowing amber eyes"),
        ("renard", "Holographic Fox", "luminescent ethereal holographic fox companion with neon data particles and glowing amber eyes"),
        ("fox", "Holographic Fox", "luminescent ethereal holographic fox companion with neon data particles and glowing amber eyes"),
        ("drone", "Tactical Companion Drone", "compact hovering reconnaissance drone with blue ring lighting and carbon-fiber thrusters"),
        ("chien", "Companion Hound", "faithful companion hound with tactical protective vest and alert posture"),
        ("dog", "Companion Hound", "faithful companion hound with tactical protective vest and alert posture"),
        ("chat", "Companion Cat", "mysterious sleek black feline companion with luminous emerald eyes"),
        ("cat", "Companion Cat", "mysterious sleek black feline companion with luminous emerald eyes"),
        ("dragon", "Ethereal Dragon Companion", "miniature ethereal dragon spirit winding through the air with shimmering scales"),
    ]
    companion = None
    for kw, name, desc in companion_candidates:
        if kw in lower:
            companion = (name, desc)
            break

    env_candidates = [
        (["ruelle", "ville néon", "ville neon", "rue", "street", "alley"], "Neon Cyberpunk Alley", "rain-slicked cyberpunk city street with glowing neon signs, wet asphalt reflections, towering skyscrapers and urban alley architecture"),
        (["temple", "sanctuaire", "sanctuary"], "Ancient Futuristic Temple", "ancient futuristic temple sanctuary with weathered stone pillars, glowing cybernetic runes, carved altars and mystical volumetric atmosphere"),
        (["chambre", "bedroom"], "Cozy Bedroom", "cozy teenage bedroom environment, bed on the right, window and desk on the left, curtains, furniture arrangement, materials, warm neutral colors and spatial layout"),
        (["forêt", "foret", "forest", "bois", "woods"], "Forest Sanctuary", "dense ancient forest environment with moss-covered trees, sunbeams filtering through the foliage and natural earthen path"),
        (["laboratoire", "lab", "recherche"], "Research Lab", "high-tech scientific laboratory interior, holographic displays, clean metal surfaces and scientific instruments"),
        (["vaisseau", "spaceship", "passerelle", "bridge"], "Starship Bridge", "starship command bridge interior, control consoles, observation viewport overlooking outer space and metallic bulkheads"),
        (["toit", "rooftop", "roof"], "Skyscraper Rooftop", "expansive skyscraper rooftop overlooking a sprawling futuristic skyline with radio antennas and dramatic skyline perspective"),
        (["bar", "lounge", "club"], "Cyberpunk Lounge", "moody cyberpunk bar interior, neon bottle displays, bar counter, atmospheric smoke and ambient lighting"),
        (["désert", "desert", "dunes"], "Dune Wasteland", "sweeping desert expanse with wind-sculpted sand dunes, distant rocky outcroppings and shimmering heat haze"),
        (["grotte", "caverne", "cave"], "Crystalline Cavern", "subterranean cavern with glowing mineral crystals, subterranean water reflections and arched rock formations"),
    ]

    detected_environments: list[tuple[str, str, int]] = []
    for kw_list, name, desc in env_candidates:
        earliest_pos = -1
        for kw in kw_list:
            pos = lower.find(kw)
            if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                earliest_pos = pos
        if earliest_pos != -1:
            detected_environments.append((name, desc, earliest_pos))

    detected_environments.sort(key=lambda x: x[2])
    environments = [(name, desc) for name, desc, _ in detected_environments]
    if not environments:
        environments = [("Environment", "cinematic setting with intricate spatial architectural details")]

    prop_candidates = [
        ("sabre", "Energy Katana", "sleek cybernetic energy katana with glowing cyan plasma edge and carbon-fiber hilt"),
        ("katana", "Energy Katana", "sleek cybernetic energy katana with glowing cyan plasma edge and carbon-fiber hilt"),
        ("écouteurs", "Wireless Headphones", "sleek premium wireless headphones resting cleanly on desk with matte finish"),
        ("headphones", "Wireless Headphones", "sleek premium wireless headphones resting cleanly on desk with matte finish"),
        ("clé", "Data Keycard", "luminescent holographic data keycard with glowing micro-circuits"),
        ("key", "Data Keycard", "luminescent holographic data keycard with glowing micro-circuits"),
        ("bouclier", "Energy Shield", "hexagonal energy barrier projector emitting a faint cyan defensive field"),
        ("grimoire", "Ancient Grimoire", "leather-bound ancient grimoire with glowing runic inscriptions across its parchment"),
    ]
    props = []
    for kw, name, desc in prop_candidates:
        if kw in lower:
            props.append((name, desc))
            break

    return {
        "primary_char": primary_char,
        "companion": companion,
        "environments": environments,
        "props": props,
    }


def extract_subjects_from_story(story: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str] | None]:
    """Compatibility wrapper returning primary character, primary environment, and prop."""
    info = extract_multi_subjects_from_story(story)
    c_name, c_desc, _ = info["primary_char"]
    e_name, e_desc = info["environments"][0]
    p_item = info["props"][0] if info["props"] else None
    return (c_name, c_desc), (e_name, e_desc), p_item


def enrich_cinematic_story_beat(
    beat: str,
    char_name: str,
    env_name: str,
    shot_index: int,
    total_shots: int,
    shot_type: str,
    camera_motion: str,
    lighting: str,
    prop_name: str | None = None,
    *,
    char_subj_idx: int = 1,
    env_subj_idx: int = 2,
    is_repeat_env: bool = False,
    companion_name: str | None = None,
    companion_subj_idx: int | None = None,
) -> str:
    """Embellish raw user story beat with rich cinematic details, atmospheric lighting, turnaround character reference locking, and per-scene decor re-use."""
    clean_beat = beat.strip().rstrip(".!?,")

    fr_replacements = [
        (r"\bexplore les ruelles d'une ville n[eé]on sous la pluie battante\b", "exploring the labyrinthine neon-lit alleys under torrential rain, water cascading from awnings"),
        (r"\bexplore les ruelles\b", "navigating narrow rain-slicked city alleys with vibrant neon reflections"),
        (r"\brencontre un renard holographique lumineux\b", "encountering a luminescent holographic fox that flickers with ambient cyan particles"),
        (r"\bqui le guide vers\b", "following the glowing companion as it leads the way toward"),
        (r"\bune porte secr[eè]te\b", "a concealed monolithic portal etched with glowing circuits"),
        (r"\bdans un temple antique futuriste\b", "inside a secluded ancient futuristic temple sanctuary"),
        (r"\bd[eé]gaine son sabre\b", "deliberately drawing an illuminated plasma katana with smooth physical weight"),
        (r"\bd[eé]fie un robot g[eé]ant\b", "standing firm in defiance against an imposing towering cybernetic automaton"),
        (r"\bse r[eé]veille\b", "stirring and opening eyes gently as morning light brightens"),
        (r"\bregarde par la fen[eê]tre\b", "turning head to gaze thoughtfully out through the rain-streaked window"),
        (r"\bprend ses [eé]couteurs\b", "reaching out to pick up headphones from the wooden desk"),
        (r"\bsort de la pi[eè]ce\b", "stepping deliberately across the room toward the exit threshold"),
    ]
    for pattern, rep in fr_replacements:
        clean_beat = re.sub(pattern, rep, clean_beat, flags=re.IGNORECASE)

    clean_stype = shot_type.split("(")[0].strip()

    if shot_index == 1:
        # Phase 1: Establishing spatial geometry, mood, lighting
        return (
            f"Begin with a wide cinematic establishing {clean_stype.lower()} of <Subject {env_subj_idx}> ({env_name}), "
            f"strictly matching the spatial architecture in <Picture {env_subj_idx}>. The environment is quiet and atmospheric "
            f"with rich volumetric {lighting.lower()} highlights and deep environmental texture. <Subject {char_subj_idx}> ({char_name}) "
            f"is present in the frame matching the turnaround character reference sheet in <Picture {char_subj_idx}>, introducing the narrative presence "
            f"with natural, understated physical posture. Hold this establishing composition briefly so the viewer assimilates the layout and mood."
        )
    elif shot_index == 2:
        # Phase 2: Focus, micro-expressions, intimate reaction
        env_clause = f"in <Subject {env_subj_idx}> (reusing the exact same environment reference <Picture {env_subj_idx}>)" if is_repeat_env else f"in <Subject {env_subj_idx}> ({env_name})"
        comp_clause = f", encountering <Subject {companion_subj_idx}> ({companion_name}, referencing turnaround sheet <Picture {companion_subj_idx}>)" if companion_name and companion_subj_idx else (f", noticing the subtle glow of <Subject 3> ({prop_name})" if prop_name else "")
        return (
            f"Make a hard cinematic cut to an intimate {clean_stype.lower()} of <Subject {char_subj_idx}> {env_clause}. Facial features, expressive gaze, "
            f"hairstyle and clothing from turnaround reference sheet <Picture {char_subj_idx}> are preserved throughout. <Subject {char_subj_idx}>{comp_clause} "
            f"reacts naturally to the surroundings, shifting gaze with subtle curiosity as {clean_beat or 'the sequence unfolds'}. Subtle micro-movements, breathing, and fabric motion bring tactile realism."
        )
    elif shot_index == total_shots:
        # Final Phase: Dramatic resolution, departure, spatial permanence
        env_clause = f"in <Subject {env_subj_idx}> (reusing the exact same environment reference <Picture {env_subj_idx}>)" if is_repeat_env else f"in <Subject {env_subj_idx}> ({env_name})"
        return (
            f"Make a hard cinematic cut to a wide tracking departure shot {env_clause}. <Subject {char_subj_idx}> (reusing turnaround reference sheet <Picture {char_subj_idx}>) "
            f"completes the movement, progressing deliberately through the space matching {clean_beat or 'the final narrative beat'}. The camera maintains steady composed "
            f"framing as <Subject {char_subj_idx}> transitions toward the edge of frame or into the deep background, leaving the enduring spatial architecture, "
            f"materials and gentle environmental particles of <Subject {env_subj_idx}> clearly visible."
        )
    else:
        # Intermediate Phases: Dynamic interaction, physical action
        env_clause = f"in <Subject {env_subj_idx}> (reusing environment reference <Picture {env_subj_idx}>)" if is_repeat_env else f"transitioning into <Subject {env_subj_idx}> ({env_name})"
        comp_clause = f" accompanied by <Subject {companion_subj_idx}> ({companion_name}, referencing <Picture {companion_subj_idx}>)" if companion_name and companion_subj_idx else (f" with <Subject 3> ({prop_name}) providing an illuminated visual focal point" if prop_name else "")
        return (
            f"Make a hard cut to a dynamic {clean_stype.lower()} framing <Subject {char_subj_idx}> (reusing turnaround reference sheet <Picture {char_subj_idx}>) "
            f"actively engaged in {clean_beat} {env_clause}{comp_clause}. Physical interaction with <Subject {env_subj_idx}> is tangible, "
            f"featuring authentic spatial depth, cast shadows, and natural weight. Camera motion executes a deliberate {camera_motion.lower()}, "
            f"tracking the character's focused movement and dramatic momentum."
        )


def generate_local_storyboard(
    *,
    story: str,
    total_duration: float,
    shot_duration: float,
    style_prompt: str,
) -> StoryboardData:
    """Generate structured MiniMax H3 StoryboardData locally with turnaround character sheets, separate empty decors, and per-scene decor re-use."""
    clean_story = (story or "").strip()
    if not clean_story:
        clean_story = (
            "A teenage anime girl wakes up in her cozy bedroom in the morning, "
            "gets up, takes her headphones from the desk and walks out."
        )

    existing = parse_existing_h3_prompt(clean_story, total_duration=total_duration, style_prompt=style_prompt)
    if existing:
        return existing

    info = extract_multi_subjects_from_story(clean_story)
    char_name, char_desc, pronoun = info["primary_char"]
    companion_item = info["companion"]
    environments = info["environments"]
    props = info["props"]

    subjects: list[StoryboardSubject] = []
    next_idx = 1

    # Subject 1: Character Turnaround Reference Sheet (multi-view on clean studio background, no scenery)
    s1_def = (
        f"{char_desc} whose facial identity, hairstyle, age appearance, body proportions, clothing "
        f"and overall character design are locked to the supplied character reference in <Picture {next_idx}>. Preserve appearance "
        f"throughout every camera cut. The background of <Picture {next_idx}> is not used."
    )
    s1_retention = (
        f"fully_preserved — preserve the exact facial identity, hairstyle, age appearance, body proportions, "
        f"clothing and recognizable character design from <Picture {next_idx}>. Allow natural changes in pose, expression and movement."
    )
    s1_img_prompt = (
        f"Character turnaround model sheet of {char_name}: multiple angles showing front view, three-quarter view, and profile side view, "
        f"clean neutral studio background, no background scenery, no environment, consistent character model design, sharp focus, masterpiece, {style_prompt}"
    )
    subjects.append(StoryboardSubject(next_idx, char_name, "character", s1_def, s1_retention, s1_img_prompt))
    char_subj_idx = next_idx
    next_idx += 1

    # Subject 2: Primary Environment (pure empty environment without people)
    env1_name, env1_desc = environments[0]
    s2_def = (
        f"the exact {env1_desc} shown in the supplied reference in <Picture {next_idx}>, including the spatial layout, materials, "
        f"furniture arrangement and architectural details. Preserve this environment and layout consistently "
        f"throughout the entire video. The character shown in <Picture {next_idx}>, if any, is ignored; <Picture 1> controls identity."
    )
    s2_retention = (
        f"fully_preserved — preserve the exact {env1_name.lower()} architecture, spatial layout, furniture, "
        f"materials, lighting direction and spatial relationships from <Picture {next_idx}>."
    )
    s2_img_prompt = (
        f"Architectural environmental concept art establishing view of {env1_name}: {env1_desc}, "
        f"wide angle view, empty space without people, spatial layout, furniture arrangement, cinematic lighting, masterpiece, sharp focus, {style_prompt}"
    )
    subjects.append(StoryboardSubject(next_idx, env1_name, "environment", s2_def, s2_retention, s2_img_prompt))
    env_subj_map = {0: next_idx}
    next_idx += 1

    # Companion Character / Secondary Character (e.g. Holographic Fox)
    comp_subj_idx = None
    if companion_item:
        comp_name, comp_desc = companion_item
        c_def = (
            f"{comp_desc} whose visual identity and proportions are locked to the turnaround reference in <Picture {next_idx}>. "
            f"Preserve recognizable appearance throughout every cut. The background of <Picture {next_idx}> is not used."
        )
        c_ret = f"fully_preserved — preserve the exact visual design and features from <Picture {next_idx}>."
        c_img = (
            f"Character turnaround model sheet of {comp_name}: multiple angles showing front view, three-quarter view, and profile view, "
            f"clean neutral studio background, no background scenery, no environment, consistent design, sharp focus, masterpiece, {style_prompt}"
        )
        subjects.append(StoryboardSubject(next_idx, comp_name, "character", c_def, c_ret, c_img))
        comp_subj_idx = next_idx
        next_idx += 1

    # Secondary Environments (e.g. Ancient Futuristic Temple)
    for e_idx in range(1, len(environments)):
        e_name, e_desc = environments[e_idx]
        e_def = (
            f"the exact {e_desc} shown in the supplied reference in <Picture {next_idx}>, including spatial layout, architecture, and lighting. "
            f"Pure empty environment without people or characters. Characters shown in <Picture {next_idx}>, if any, are ignored; <Picture 1> controls character identity."
        )
        e_ret = f"fully_preserved — preserve the exact {e_name.lower()} architecture, spatial layout, materials, and lighting direction from <Picture {next_idx}>."
        e_img = (
            f"Architectural environmental concept art establishing view of {e_name}: {e_desc}, "
            f"wide angle view, pure empty space without any people or characters, spatial layout, cinematic lighting, masterpiece, sharp focus, {style_prompt}"
        )
        subjects.append(StoryboardSubject(next_idx, e_name, "environment", e_def, e_ret, e_img))
        env_subj_map[e_idx] = next_idx
        next_idx += 1

    # Props
    prop_name = None
    if props:
        p_name, p_desc = props[0]
        prop_name = p_name
        p_def = f"{p_desc} locked to supplied reference in <Picture {next_idx}>. Preserve recognizable design throughout every cut."
        p_ret = f"fully_preserved — preserve the exact visual design from <Picture {next_idx}>."
        p_img = f"Concept art prop reference sheet of {p_name}: {p_desc}, isolated on clean background, masterpiece, {style_prompt}"
        subjects.append(StoryboardSubject(next_idx, p_name, "prop", p_def, p_ret, p_img))
        next_idx += 1

    raw_beats = [s.strip() for s in re.split(r'[.!?;\n]', clean_story) if s.strip()]
    if not raw_beats:
        raw_beats = [clean_story]

    # Calculate dynamic cinematic durations respecting shot_duration as an upper bound ceiling
    shot_durations = calculate_cinematic_shot_durations(
        total_duration=total_duration,
        max_shot_duration=shot_duration,
        num_beats=len(raw_beats),
    )
    num_shots = len(shot_durations)

    scenes: list[StoryboardScene] = []
    current_time = 0.0
    active_env_idx = 0
    used_envs = set()

    for i, duration in enumerate(shot_durations):
        idx = i + 1
        stype, shot_desc = SHOT_TYPES[min(i, len(SHOT_TYPES) - 1)]
        cam = CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)]
        lighting = LIGHTING_MOODS[i % len(LIGHTING_MOODS)]
        ts_str = format_timestamp(current_time)

        beat = raw_beats[i % len(raw_beats)]
        lower_beat = beat.lower()

        # Check if beat specifically names one of the secondary environments
        for env_i, (e_name, e_desc) in enumerate(environments):
            e_kw = e_name.lower().split()[0]
            if e_kw in lower_beat:
                active_env_idx = env_i
                break
        else:
            if len(environments) > 1 and i >= num_shots // 2 and active_env_idx == 0:
                active_env_idx = 1

        curr_env_name, _ = environments[active_env_idx]
        curr_env_subj_idx = env_subj_map.get(active_env_idx, 2)
        is_repeat_env = curr_env_subj_idx in used_envs
        used_envs.add(curr_env_subj_idx)

        prompt_content = enrich_cinematic_story_beat(
            beat=beat,
            char_name=char_name,
            env_name=curr_env_name,
            shot_index=idx,
            total_shots=num_shots,
            shot_type=stype,
            camera_motion=cam,
            lighting=lighting,
            prop_name=prop_name,
            char_subj_idx=char_subj_idx,
            env_subj_idx=curr_env_subj_idx,
            is_repeat_env=is_repeat_env,
            companion_name=companion_item[0] if companion_item else None,
            companion_subj_idx=comp_subj_idx,
        )

        full_prompt = (
            f"[Camera: {cam.split('(')[-1].rstrip(')')}] {prompt_content} "
            f"Shot: {stype}. Lighting: {lighting}. Style: {style_prompt}."
        )

        scenes.append(
            StoryboardScene(
                index=idx,
                title=f"Plan {idx} : {stype.split('(')[0].strip()}",
                duration=duration,
                start_time=current_time,
                timestamp_str=ts_str,
                shot_type=stype,
                camera_motion=cam,
                lighting=lighting,
                visual_prompt=prompt_content,
                full_prompt=full_prompt,
                image_prompt=prompt_content,
                continuity_visual=(i > 0),
                continuity_audio=(i > 0),
            )
        )
        current_time += duration
    summary = (
        f"[reference generation] A cinematic narrative sequence set in <Subject 2> featuring <Subject 1>. "
        f"The visual story unfolds with purposeful camera framing and deliberate pacing while preserving identity and environmental continuity."
    )
    intro = (
        f"The target video is a polished cinematic sequence. Movement is natural, understated and physically believable. "
        f"The environment layout is fixed throughout: spatial relationships established in <Picture 2> remain consistent. "
        f"H3 uses deliberate cinematic cuts between clearly distinct viewpoints, each introducing meaningful framing and visual information."
    )
    soundscape = (
        f"Detailed environmental acoustics and ambient audio matching <Subject 2> throughout. "
        f"Subtle fabric movement, natural footsteps on surfaces, gentle atmospheric airflow, and nuanced room reverb."
    )

    full_prompt = format_h3_structured_prompt(subjects, summary, intro, scenes, soundscape)

    return StoryboardData(
        subjects=subjects,
        scenes=scenes,
        summary=summary,
        detailed_description_intro=intro,
        soundscape=soundscape,
        music="N/A",
        full_h3_prompt=full_prompt,
    )


def generate_llm_storyboard(
    *,
    story: str,
    total_duration: float,
    shot_duration: float,
    style_prompt: str,
    llm_url: str = "http://127.0.0.1:11434",
    llm_model: str = "qwen2.5:7b",
    api_format: str = "ollama",
    timeout: int = 40,
) -> StoryboardData | None:
    """Generate structured MiniMax H3 StoryboardData using local LLM with narrative embellishment."""
    # shot_duration is the MAXIMUM allowed duration per shot
    min_shots = max(1, math.ceil(total_duration / max(2.0, shot_duration)))
    target_shots = max(min_shots, min(int(total_duration // 2.5), 6))

    system_prompt = (
        "You are an award-winning Hollywood film director and master cinematic prompt architect for MiniMax H3.\n"
        "Your task is to EMBELLISH and expand the user's narrative story into MiniMax H3's official multi-subject cinematic reference format.\n\n"
        "CRITICAL RULES:\n"
        "1. EMBELLISH THE STORY: Do not just repeat the user's text. Add vivid sensory details, lighting atmosphere, facial micro-expressions, posture, and spatial staging.\n"
        "2. CHARACTER TURNAROUND SHEETS: Character reference image prompts MUST specify multi-angle turnaround model sheets (front view, three-quarter view, profile side view, clean neutral studio background, no background scenery, no environment).\n"
        "3. SEPARATE DECOR REFERENCES: Environment reference image prompts MUST specify pure empty environments without any people or characters, showing spatial architecture and layout.\n"
        "4. REUSE REFERENCES ACROSS SCENES: Re-use the exact same <Picture 1> across all scenes for the character. Re-use <Picture 2> (or subsequent decor) across all scenes set in that location. Never duplicate references for the same subject.\n"
        f"5. DYNAMIC SHOT DURATIONS: Assign dynamic durations (between 2.5s and {shot_duration}s) respecting shot_duration as a maximum ceiling. Sum must strictly equal {total_duration}s.\n"
        "6. STRICT SUBJECT TAGGING: Always use <Subject 1>, <Subject 2>, <Picture 1>, <Picture 2> in definitions and shot descriptions.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "subject_1": {\n'
        '    "name": "Character Name",\n'
        '    "definition": "Detailed character identity locked to Picture 1",\n'
        '    "image_prompt": "Clean T2I prompt for character design concept reference sheet"\n'
        "  },\n"
        '  "subject_2": {\n'
        '    "name": "Environment Name",\n'
        '    "definition": "Detailed architectural layout and spatial geometry locked to Picture 2",\n'
        '    "image_prompt": "Clean T2I prompt for empty environment architecture layout"\n'
        "  },\n"
        '  "summary": "Short cinematic summary across Subject 1 and Subject 2",\n'
        '  "detailed_description_intro": "Cinematic motion guidelines",\n'
        '  "shots": [\n'
        "    {\n"
        '      "shot_index": 1,\n'
        '      "duration": 4.5,\n'
        '      "timestamp_str": "00:00.000",\n'
        '      "description": "Begin with wide establishing view of <Subject 2>..."\n'
        "    }\n"
        "  ],\n"
        '  "soundscape": "Environmental ambient soundscape details",\n'
        '  "music": "N/A"\n'
        "}"
    )

    user_message = (
        f"User Narrative Script to Adapt and Embellish:\n{story}\n\n"
        f"Production Specifications:\n"
        f"- Target number of shots: {target_shots}\n"
        f"- Maximum duration per shot: {shot_duration}s (Each shot <= {shot_duration}s)\n"
        f"- Total duration: {total_duration}s (Sum of all shot durations must equal {total_duration}s)\n"
        f"- Visual aesthetic: {style_prompt}"
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

        clean_content = re.sub(r"^```(?:json)?\s*", "", content_str.strip())
        clean_content = re.sub(r"\s*```$", "", clean_content).strip()
        parsed = json.loads(clean_content)

        s1_data = parsed.get("subject_1") or {}
        s2_data = parsed.get("subject_2") or {}
        s1_name = str(s1_data.get("name") or "Protagonist")
        s2_name = str(s2_data.get("name") or "Environment")

        s1_def = str(s1_data.get("definition") or f"{s1_name} locked to <Picture 1>.")
        s2_def = str(s2_data.get("definition") or f"{s2_name} locked to <Picture 2>.")
        s1_img = str(s1_data.get("image_prompt") or f"Character concept art portrait of {s1_name}, {style_prompt}")
        s2_img = str(s2_data.get("image_prompt") or f"Architectural environment concept art of {s2_name}, {style_prompt}")

        subjects = [
            StoryboardSubject(
                1, s1_name, "character", s1_def,
                f"fully_preserved — preserve recognizable design from <Picture 1>.",
                s1_img
            ),
            StoryboardSubject(
                2, s2_name, "environment", s2_def,
                f"fully_preserved — preserve architecture and spatial layout from <Picture 2>.",
                s2_img
            ),
        ]

        raw_shots = parsed.get("shots") or []
        num_raw = len(raw_shots) or target_shots

        # Parse or compute dynamic shot durations
        parsed_durations = [float(s.get("duration") or 0.0) for s in raw_shots if isinstance(s, dict)]
        if len(parsed_durations) == len(raw_shots) and all(0.5 < d <= shot_duration + 0.5 for d in parsed_durations):
            # Scale to match total_duration exactly
            scale = total_duration / max(0.1, sum(parsed_durations))
            durations = [round(d * scale, 1) for d in parsed_durations]
            diff = round(total_duration - sum(durations), 1)
            durations[-1] = round(durations[-1] + diff, 1)
        else:
            durations = calculate_cinematic_shot_durations(total_duration, shot_duration, num_raw)

        scenes: list[StoryboardScene] = []
        curr_t = 0.0

        for i, s_dur in enumerate(durations):
            idx = i + 1
            s_raw = raw_shots[i] if i < len(raw_shots) else {}
            ts_str = format_timestamp(curr_t)
            desc = str(s_raw.get("description") or f"Scene {idx}")
            scenes.append(
                StoryboardScene(
                    index=idx,
                    title=f"Plan {idx}",
                    start_time=curr_t,
                    duration=s_dur,
                    timestamp_str=ts_str,
                    shot_type=SHOT_TYPES[min(i, len(SHOT_TYPES) - 1)][0],
                    camera_motion=CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)],
                    lighting=LIGHTING_MOODS[i % len(LIGHTING_MOODS)],
                    visual_prompt=desc,
                    full_prompt=desc,
                    image_prompt=desc,
                    continuity_visual=(i > 0),
                    continuity_audio=(i > 0),
                )
            )
            curr_t += s_dur

        summary = str(parsed.get("summary") or f"[reference generation] Sequence in <Subject 2> featuring <Subject 1>.")
        intro = str(parsed.get("detailed_description_intro") or "Cinematic motion.")
        soundscape = str(parsed.get("soundscape") or "Environmental soundscape.")
        music = str(parsed.get("music") or "N/A")

        full_prompt = format_h3_structured_prompt(subjects, summary, intro, scenes, soundscape, music)

        return StoryboardData(
            subjects=subjects,
            scenes=scenes,
            summary=summary,
            detailed_description_intro=intro,
            soundscape=soundscape,
            music=music,
            full_h3_prompt=full_prompt,
        )
    except Exception as exc:
        log.warning("LLM Storyboard Generation failed (%s). Falling back to rule-based engine.", exc)
        return None


def generate_local_scenes(
    *,
    story: str,
    num_scenes: int,
    total_duration: float,
    style_prompt: str,
) -> list[StoryboardScene]:
    """Compatibility wrapper returning list of StoryboardScene."""
    shot_dur = total_duration / max(1, num_scenes)
    sb_data = generate_local_storyboard(
        story=story,
        total_duration=total_duration,
        shot_duration=shot_dur,
        style_prompt=style_prompt,
    )
    return sb_data.scenes


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
    """Compatibility wrapper returning list of StoryboardScene."""
    shot_dur = total_duration / max(1, num_scenes)
    sb_data = generate_llm_storyboard(
        story=story,
        total_duration=total_duration,
        shot_duration=shot_dur,
        style_prompt=style_prompt,
        llm_url=llm_url,
        llm_model=llm_model,
        api_format=api_format,
        timeout=timeout,
    )
    return sb_data.scenes if sb_data else None


def generate_subject_reference_images(
    subjects: list[StoryboardSubject],
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
    """Generate canonical AI reference images for each Subject (<Picture 1>, <Picture 2>)."""
    generated_tensors: list[torch.Tensor] = []
    b64_list: list[str] = []

    has_diffusion = model is not None and clip is not None and vae is not None
    ext_count = int(external_images.shape[0]) if external_images is not None and external_images.ndim == 4 else 0

    for i, subj in enumerate(subjects):
        img_tensor: torch.Tensor | None = None

        # Option A: Real Diffusion AI Image Generation
        if has_diffusion:
            try:
                comfy_core = get_comfy_core()
                if comfy_core:
                    encoder = comfy_core.CLIPTextEncode()
                    pos_cond = encoder.encode(clip, subj.image_prompt)[0]
                    neg_cond = encoder.encode(clip, negative_prompt)[0]

                    latent_channels = 4
                    if hasattr(model, "model") and hasattr(model.model, "latent_format"):
                        latent_channels = getattr(model.model.latent_format, "latent_channels", 4) or 4
                    elif hasattr(model, "latent_format"):
                        latent_channels = getattr(model.latent_format, "latent_channels", 4) or 4

                    latent_w = max(4, width // 8)
                    latent_h = max(4, height // 8)
                    empty_latent = {"samples": torch.zeros([1, latent_channels, latent_h, latent_w])}

                    sampled = comfy_core.common_ksampler(
                        model=model,
                        seed=seed + i * 37,
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
                    log.info("Story Director: Generated Reference Image for %s (%s)", subj.subject_tag, subj.name)
            except Exception as diff_exc:
                log.warning("Diffusion reference generation failed for %s: %s", subj.subject_tag, diff_exc)
                img_tensor = None

        # Option B: External supplied image batch
        if img_tensor is None and ext_count > 0:
            pick_idx = min(i, ext_count - 1)
            img_tensor = external_images[pick_idx : pick_idx + 1]

        # Option C: Stylized Visual Concept Art fallback
        if img_tensor is None:
            img_tensor = render_storyboard_visual_concept(
                scene_num=subj.index,
                title=subj.name,
                story_snippet=subj.definition,
                lighting="Atmospheric",
                category=subj.category,
                width=width,
                height=height,
                seed=seed,
            )

        subj.image_tensor = img_tensor
        b64_str = tensor_to_b64_jpeg(img_tensor)
        subj.image_b64 = b64_str

        # Save reference image to ComfyUI input folder for persistent file loading and preview in Director
        try:
            import folder_paths
            input_dir = folder_paths.get_input_directory()
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', subj.name).strip('_').lower() or f"subj_{subj.index}"
            fname = f"story_ref_{subj.index}_{safe_name}.png"
            fpath = os.path.join(input_dir, fname)
            t_slice = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
            arr = (t_slice.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(fpath, format="PNG")
            subj.image_file = fname
            log.info("Story Director: Saved reference image to %s", fpath)
        except Exception as save_exc:
            log.debug("Notice: Could not save reference image to input folder: %s", save_exc)
            subj.image_file = ""
        generated_tensors.append(img_tensor)
        b64_list.append(b64_str)

    batch_tensor = torch.cat(generated_tensors, dim=0) if generated_tensors else torch.zeros([1, height, width, 3])
    return batch_tensor, b64_list


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
    """Compatibility wrapper generating reference images for a list of scenes."""
    dummy_subjects = [
        StoryboardSubject(
            index=s.index,
            name=s.title,
            category="character" if s.index == 1 else "environment",
            definition=s.visual_prompt,
            retention="fully_preserved",
            image_prompt=s.image_prompt or s.visual_prompt,
        )
        for s in scenes
    ]
    batch_tensor, b64_list = generate_subject_reference_images(
        dummy_subjects,
        width=width,
        height=height,
        model=model,
        clip=clip,
        vae=vae,
        steps=steps,
        cfg=cfg,
        seed=seed,
        negative_prompt=negative_prompt,
        external_images=external_images,
    )
    for i, s in enumerate(scenes):
        s.image_tensor = dummy_subjects[i].image_tensor
        s.image_b64 = dummy_subjects[i].image_b64
        s.image_file = getattr(dummy_subjects[i], "image_file", "")
    return batch_tensor, b64_list


def build_director_timeline(
    scenes: list[StoryboardScene],
    *,
    subjects: list[StoryboardSubject] | None = None,
    full_h3_prompt: str = "",
    director_mode: str = "mixed",
    width: int = 864,
    height: int = 480,
    fps: float = 24.0,
    ref_max_size: int = 864,
) -> dict[str, Any]:
    """Assemble a compliant MiniMax H3 Motion Director timeline specification using canonical references."""
    import copy

    total_frames = sum(max(5, int(round(s.duration * fps))) for s in scenes)

    # Reference pictures from subjects or scenes
    ref_pictures: list[dict[str, Any]] = []
    if subjects:
        for subj in subjects:
            if subj.image_b64 or getattr(subj, "image_file", ""):
                idx = len(ref_pictures)
                ref_pictures.append({
                    "index": idx,
                    "slot": idx,
                    "imageFile": getattr(subj, "image_file", "") or "",
                    "fileName": getattr(subj, "image_file", "") or "",
                    "imageB64": subj.image_b64 or "",
                    "assetId": f"subject-{subj.index}",
                    "name": subj.name,
                    "type": "input",
                })
    if scenes:
        for s in scenes:
            if s.image_b64 or getattr(s, "image_file", ""):
                scene_name = s.title or f"Plan {s.index}"
                if not any(r.get("name") == scene_name for r in ref_pictures):
                    idx = len(ref_pictures)
                    ref_pictures.append({
                        "index": idx,
                        "slot": idx,
                        "imageFile": getattr(s, "image_file", "") or "",
                        "fileName": getattr(s, "image_file", "") or "",
                        "imageB64": s.image_b64 or "",
                        "assetId": f"scene-{s.index}",
                        "name": scene_name,
                        "type": "input",
                    })

    effective_prompt = full_h3_prompt or (scenes[0].full_prompt if scenes else "")

    if director_mode.lower() == "mixed":
        mixed_segments = []
        for s in scenes:
            seg_dict: dict[str, Any] = {
                "id": f"seg_{s.index}",
                "mode": "r2v",
                "prompt": s.full_prompt,
                "duration": float(s.duration),
                "inputs": {
                    "resultRefs": [],
                    "pictures": [copy.deepcopy(r) for r in ref_pictures],
                    "identityPictures": [copy.deepcopy(r) for r in ref_pictures],
                    "startFrame": {
                        "imageFile": getattr(s, "image_file", "") or (ref_pictures[0]["imageFile"] if ref_pictures else ""),
                        "imageB64": s.image_b64 or (ref_pictures[0]["imageB64"] if ref_pictures else ""),
                    },
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
            "global": {
                "taskType": "r2v",
                "prompt": effective_prompt,
                "refs": [copy.deepcopy(r) for r in ref_pictures],
            },
            "segments": mixed_segments,
        }
        return timeline_dict

    # Standard / Global timeline structure
    cursor = 0
    std_segments = []
    task_key = "r2v" if (director_mode in ("r2v", "mixed") or ref_pictures) else director_mode
    is_batch_task = task_key in ("r2v", "t2v", "i2v", "r2i", "t2i", "i2i")

    for s in scenes:
        seg_frames = max(5, int(round(s.duration * fps)))
        seg_refs: list[dict[str, Any]] = []
        for r in ref_pictures:
            r_copy = copy.deepcopy(r)
            r_copy["assetId"] = f"segment-s{s.index - 1}-picture-{r['index'] + 1}"
            seg_refs.append(r_copy)

        if s.image_b64 or getattr(s, "image_file", ""):
            scene_name = s.title or f"Plan {s.index}"
            if not any(r.get("name") == scene_name for r in seg_refs):
                seg_refs.append({
                    "index": len(seg_refs),
                    "slot": len(seg_refs),
                    "imageFile": getattr(s, "image_file", "") or "",
                    "fileName": getattr(s, "image_file", "") or "",
                    "imageB64": s.image_b64 or "",
                    "assetId": f"segment-s{s.index - 1}-scene-{s.index}",
                    "name": scene_name,
                    "type": "input",
                })
        std_segments.append(
            {
                "id": f"s{s.index - 1}",
                "start": cursor,
                "length": seg_frames,
                "durationSec": float(s.duration),
                "frameCount": seg_frames,
                "prompt": s.full_prompt,
                "negativePrompt": "",
                "taskType": task_key,
                "refs": seg_refs,
                "useCommonAssets": True,
                "excludedCommonAssetIds": [],
                "referenceVideo": {},
            }
        )
        cursor += seg_frames

    return {
        "version": 4,
        "timelineMode": "prompt_batch" if is_batch_task else "video",
        "editMode": "segment" if is_batch_task else "global",
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
        "global": {
            "taskType": task_key,
            "prompt": effective_prompt,
            "refs": [
                {
                    **copy.deepcopy(r),
                    "assetId": f"global-picture-{r['index'] + 1}",
                }
                for r in ref_pictures
            ],
        },
        "r2vCommon": {
            "refs": [
                {
                    **copy.deepcopy(r),
                    "assetId": f"common-picture-{r['index'] + 1}",
                }
                for r in ref_pictures
            ],
            "refVideos": [],
            "refAudios": [],
        },
        "segments": std_segments,
    }


def format_script_summary(
    scenes: list[StoryboardScene],
    total_duration: float,
    title: str = "",
    full_h3_prompt: str = "",
) -> str:
    """Format human-readable script breakdown (returns full MiniMax H3 prompt if present)."""
    if full_h3_prompt and "subject_definitions:" in full_h3_prompt:
        return full_h3_prompt

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
        lines.append(f"  • Prompt Vidéo : {s.full_prompt}")
        lines.append(f"  • Prompt Image : {s.image_prompt}")
        lines.append(f"  • Continuité: Visuelle = {'Oui' if s.continuity_visual else 'Non'}, Audio = {'Oui' if s.continuity_audio else 'Non'}")
        lines.append("")
    return "\n".join(lines)
