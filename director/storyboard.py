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
    """Render a stylized visual concept art illustration (character portrait or environment scenery)."""
    rng = random.Random(seed + scene_num * 101)
    img = Image.new("RGB", (width, height), color=(10, 14, 22))
    draw = ImageDraw.Draw(img)

    lower_snippet = (story_snippet + " " + lighting + " " + title).lower()
    is_cyber = any(k in lower_snippet for k in ["neon", "cyber", "robot", "pluie", "rain", "tokyo", "futur"])
    is_sunset = any(k in lower_snippet for k in ["soleil", "sunset", "golden", "or", "nature", "forêt"])

    if category == "character":
        # Character reference sheet illustration
        cx = width // 2
        cy = int(height * 0.52)
        for y in range(height):
            ratio = y / max(1, height)
            r = int(12 + ratio * 24)
            g = int(14 + ratio * 26)
            b = int(24 + ratio * 45)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        halo_rad = int(min(width, height) * 0.38)
        glow_col = (0, 200, 255) if is_cyber else ((255, 170, 60) if is_sunset else (140, 180, 255))
        for r_step in range(halo_rad, 0, -8):
            af = (1.0 - r_step / halo_rad) ** 1.8
            draw.ellipse(
                [(cx - r_step, cy - int(r_step * 1.1)), (cx + r_step, cy + int(r_step * 0.9))],
                fill=(int(glow_col[0] * af), int(glow_col[1] * af), int(glow_col[2] * af)),
            )

        head_r = int(min(width, height) * 0.12)
        head_y = int(cy - head_r * 1.8)
        draw.ellipse([(cx - head_r, head_y - head_r), (cx + head_r, head_y + head_r)], fill=(14, 18, 28))
        draw.polygon(
            [
                (cx - int(head_r * 2.4), height),
                (cx - int(head_r * 1.5), head_y + head_r),
                (cx + int(head_r * 1.5), head_y + head_r),
                (cx + int(head_r * 2.4), height),
            ],
            fill=(14, 18, 28),
        )
        draw.arc(
            [(cx - head_r, head_y - head_r), (cx + head_r, head_y + head_r)],
            start=160,
            end=330,
            fill=(0, 230, 255) if is_cyber else (255, 210, 120),
            width=3,
        )
    else:
        # Environment architectural layout illustration
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

        # Room / architecture silhouette
        desk_x2 = int(width * 0.42)
        draw.rectangle([(int(width * 0.08), int(height * 0.22)), (desk_x2, horizon_y)], fill=(16, 20, 30))
        bed_x1 = int(width * 0.58)
        draw.rectangle([(bed_x1, int(height * 0.35)), (int(width * 0.92), horizon_y)], fill=(20, 24, 34))

        for y in range(horizon_y, height):
            ratio = (y - horizon_y) / max(1, height - horizon_y)
            draw.line([(0, y), (width, y)], fill=(int(14 + ratio * 18), int(16 + ratio * 18), int(28 + ratio * 32)))

    # Frame lines and cinematic label
    draw.rectangle([(16, 16), (width - 17, height - 17)], outline=(35, 55, 80), width=1)
    tag = f"<Subject {scene_num}> : {title} [{'Character Reference' if category == 'character' else 'Environment Reference'}]"
    draw.text((32, 28), tag, fill=(0, 220, 250))

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


def extract_subjects_from_story(story: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str] | None]:
    """Analyze high-level narrative to extract Character (Subject 1) and Environment (Subject 2)."""
    lower = story.lower()
    char_name = "Protagonist"
    char_desc = "cinematic character"
    char_keywords = [
        ("chevalier cyberpunk", "Cyberpunk Knight", "armored cyberpunk knight in high-tech mechanical plate armor with glowing LED accents, cybernetic visor and utility cloak"),
        ("cyberpunk knight", "Cyberpunk Knight", "armored cyberpunk knight in high-tech mechanical plate armor with glowing LED accents, cybernetic visor and utility cloak"),
        ("fille", "Girl", "teenage anime girl with expressive facial identity, distinctive hairstyle, school or casual clothing, realistic proportions"),
        ("girl", "Girl", "teenage anime girl with expressive facial identity, distinctive hairstyle, school or casual clothing, realistic proportions"),
        ("femme", "Woman", "young adult woman with elegant features, distinctive hairstyle, tailored modern clothing and natural poise"),
        ("woman", "Woman", "young adult woman with elegant features, distinctive hairstyle, tailored modern clothing and natural poise"),
        ("homme", "Man", "young adult man with rugged facial features, styled hair, atmospheric jacket and clothing"),
        ("man", "Man", "young adult man with rugged facial features, styled hair, atmospheric jacket and clothing"),
        ("ninja", "Ninja", "covert shinobi ninja with dark tactical shinobi gear, cloth mask, strapped gauntlets and stealth attire"),
        ("samourai", "Samurai", "ronin samurai warrior in traditional armored lacquered plates with patterned haori and topknot"),
        ("samurai", "Samurai", "ronin samurai warrior in traditional armored lacquered plates with patterned haori and topknot"),
        ("sorcier", "Wizard", "arcane wizard in embroidered mystical robes with runic leather wraps and glowing amulet"),
        ("wizard", "Wizard", "arcane wizard in embroidered mystical robes with runic leather wraps and glowing amulet"),
        ("robot", "Robot", "humanoid android robot with polished chassis, optic sensors, exposed joint hydraulics and sleek design"),
        ("guerrier", "Warrior", "battle-hardened warrior with combat armor, survival harness and determined expression"),
        ("warrior", "Warrior", "battle-hardened warrior with combat armor, survival harness and determined expression"),
    ]
    for kw, name, desc in char_keywords:
        if kw in lower:
            char_name, char_desc = name, desc
            break

    env_name = "Environment"
    env_desc = "cinematic setting"
    env_keywords = [
        ("chambre", "Bedroom", "cozy teenage bedroom environment, bed, window, curtains, desk, furniture arrangement, materials, warm neutral colors and spatial layout"),
        ("bedroom", "Bedroom", "cozy teenage bedroom environment, bed, window, curtains, desk, furniture arrangement, materials, warm neutral colors and spatial layout"),
        ("ruelle", "Neon City Street", "rain-slicked cyberpunk city street with neon signs, wet asphalt reflections, towering skyscrapers and urban alley architecture"),
        ("neon", "Neon City Street", "rain-slicked cyberpunk city street with neon signs, wet asphalt reflections, towering skyscrapers and urban alley architecture"),
        ("street", "City Street", "cinematic urban street environment, architectural storefronts, pavement, streetlights and atmospheric perspective"),
        ("temple", "Ancient Temple", "ancient futuristic temple sanctuary with weathered stone pillars, glowing cybernetic runes, carved altars and mystical volumetric atmosphere"),
        ("forêt", "Forest Sanctuary", "dense ancient forest environment with moss-covered trees, sunbeams filtering through the foliage and natural earthen path"),
        ("forest", "Forest Sanctuary", "dense ancient forest environment with moss-covered trees, sunbeams filtering through the foliage and natural earthen path"),
        ("laboratoire", "Research Lab", "high-tech scientific laboratory interior, holographic displays, clean metal surfaces and scientific instruments"),
        ("lab", "Research Lab", "high-tech scientific laboratory interior, holographic displays, clean metal surfaces and scientific instruments"),
        ("vaisseau", "Spaceship Bridge", "starship command bridge interior, control consoles, observation viewport overlooking outer space and metallic bulkheads"),
        ("spaceship", "Spaceship Bridge", "starship command bridge interior, control consoles, observation viewport overlooking outer space and metallic bulkheads"),
    ]
    for kw, name, desc in env_keywords:
        if kw in lower:
            env_name, env_desc = name, desc
            break

    prop_item = None
    if "renard" in lower or "fox" in lower:
        prop_item = ("Holographic Fox", "luminescent ethereal holographic fox companion with neon data particles and glowing amber eyes")

    return (char_name, char_desc), (env_name, env_desc), prop_item


def generate_local_storyboard(
    *,
    story: str,
    total_duration: float,
    shot_duration: float,
    style_prompt: str,
) -> StoryboardData:
    """Generate structured MiniMax H3 StoryboardData locally using rule-based narrative decomposition."""
    clean_story = (story or "").strip()
    if not clean_story:
        clean_story = "A teenage anime girl wakes up in her cozy bedroom in the morning, gets up, takes her headphones from the desk and walks out."

    existing = parse_existing_h3_prompt(clean_story, total_duration=total_duration, style_prompt=style_prompt)
    if existing:
        return existing

    (char_name, char_desc), (env_name, env_desc), prop_item = extract_subjects_from_story(clean_story)

    s1_def = (
        f"{char_desc} whose facial identity, hairstyle, age appearance, body proportions, clothing "
        f"and overall character design are locked to the supplied character reference. Preserve appearance "
        f"throughout every camera cut. The background of <Picture 1> is not used."
    )
    s1_retention = (
        f"fully_preserved — preserve the exact facial identity, hairstyle, age appearance, body proportions, "
        f"clothing and recognizable character design from <Picture 1>. Allow natural changes in pose, expression and movement."
    )
    s1_img_prompt = (
        f"Character design concept art reference sheet portrait and full body view of {char_name}: {char_desc}, "
        f"isolated, clean background, neutral lighting, masterpiece, sharp focus, {style_prompt}"
    )

    s2_def = (
        f"the exact {env_desc} shown in the supplied reference, including the spatial layout, materials, "
        f"furniture arrangement and architectural details. Preserve this environment and layout consistently "
        f"throughout the entire video. The character shown in <Picture 2>, if any, is ignored; <Picture 1> controls identity."
    )
    s2_retention = (
        f"fully_preserved — preserve the exact {env_name.lower()} architecture, spatial layout, furniture, "
        f"materials, lighting direction and spatial relationships from <Picture 2>."
    )
    s2_img_prompt = (
        f"Architectural environmental concept art establishing view of {env_name}: {env_desc}, "
        f"wide angle view, empty space without people, spatial layout, furniture arrangement, cinematic lighting, masterpiece, sharp focus, {style_prompt}"
    )

    subjects = [
        StoryboardSubject(1, char_name, "character", s1_def, s1_retention, s1_img_prompt),
        StoryboardSubject(2, env_name, "environment", s2_def, s2_retention, s2_img_prompt),
    ]

    if prop_item:
        p_name, p_desc = prop_item
        p_def = f"{p_desc} locked to supplied reference. Preserve recognizable design throughout every cut."
        p_ret = f"fully_preserved — preserve the exact visual design from <Picture 3>."
        p_img = f"Concept art prop reference sheet of {p_name}: {p_desc}, isolated on clean background, masterpiece, {style_prompt}"
        subjects.append(StoryboardSubject(3, p_name, "prop", p_def, p_ret, p_img))

    # Calculate shots
    num_shots = max(1, int(round(total_duration / max(1.5, shot_duration))))
    time_per_shot = round(total_duration / num_shots, 2)

    raw_beats = [s.strip() for s in re.split(r"[.!?;\n]+", clean_story) if s.strip()]
    if not raw_beats:
        raw_beats = [clean_story]

    scenes: list[StoryboardScene] = []
    current_time = 0.0

    for i in range(num_shots):
        idx = i + 1
        stype, shot_desc = SHOT_TYPES[min(i, len(SHOT_TYPES) - 1)]
        cam = CAMERA_MOTIONS[i % len(CAMERA_MOTIONS)]
        lighting = LIGHTING_MOODS[i % len(LIGHTING_MOODS)]
        ts_str = format_timestamp(current_time)

        if i == 0:
            prompt_content = f"Begin with a {stype.split('(')[0].strip().lower()} of <Subject 2>, matching the layout in <Picture 2>. <Subject 1> is present in the scene, introducing the narrative setting."
        elif i == 1:
            prompt_content = f"a medium close-up of <Subject 1> in <Subject 2>, capturing natural movement and facial expression."
        elif i == 2:
            prompt_content = f"a three-quarter cinematic angle as <Subject 1> interacts deliberately with the environment in <Subject 2>."
        elif i == num_shots - 1:
            prompt_content = f"a concluding wide shot showing <Subject 1> completing the action and moving through <Subject 2>."
        else:
            beat = raw_beats[i % len(raw_beats)]
            prompt_content = f"a {stype.split('(')[0].strip().lower()} framing <Subject 1> engaged in {beat} within <Subject 2>."

        full_prompt = (
            f"[Camera: {cam.split('(')[-1].rstrip(')')}] {prompt_content} "
            f"Shot: {stype}. Lighting: {lighting}. Style: {style_prompt}."
        )

        scenes.append(
            StoryboardScene(
                index=idx,
                title=f"Plan {idx} : {stype.split('(')[0].strip()}",
                duration=time_per_shot,
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
        current_time += time_per_shot

    summary = f"[reference generation] A sequence set in <Subject 2>. <Subject 1> navigates the environment with deliberate cinematic cuts while preserving the same character and setting."
    intro = (
        f"The target video is a polished cinematic sequence. Movement is natural, understated and physically believable. "
        f"The environment layout is fixed throughout: the spatial relationships established in <Picture 2> remain consistent. "
        f"H3 should use real cinematic cuts between clearly different viewpoints; each cut introduces meaningful new framing or visual information."
    )
    soundscape = (
        f"Atmospheric environmental soundscape matching <Subject 2> throughout. Subtle movement, footsteps, and natural room acoustics."
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
    """Generate structured MiniMax H3 StoryboardData using local LLM."""
    num_shots = max(1, int(round(total_duration / max(1.5, shot_duration))))
    system_prompt = (
        "You are an award-winning cinematic director and AI video prompt engineer for MiniMax H3.\n"
        "Your task is to decompose the user's narrative story into MiniMax H3's official multi-subject cinematic reference format.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "subject_1": {\n'
        '    "name": "Girl",\n'
        '    "definition": "Detailed character identity locked to Picture 1",\n'
        '    "image_prompt": "Clean T2I prompt for character design reference sheet"\n'
        "  },\n"
        '  "subject_2": {\n'
        '    "name": "Bedroom",\n'
        '    "definition": "Detailed environment architecture and layout locked to Picture 2",\n'
        '    "image_prompt": "Clean T2I prompt for empty environment architecture layout"\n'
        "  },\n"
        '  "summary": "Short narrative summary across Subject 1 and Subject 2",\n'
        '  "detailed_description_intro": "Cinematic motion guidelines",\n'
        '  "shots": [\n'
        "    {\n"
        '      "shot_index": 1,\n'
        '      "timestamp_str": "00:00.000",\n'
        '      "description": "Begin with wide establishing view of <Subject 2> with <Subject 1>"\n'
        "    }\n"
        "  ],\n"
        '  "soundscape": "Environmental ambient soundscape",\n'
        '  "music": "N/A"\n'
        "}"
    )

    user_message = (
        f"Story to adapt:\n{story}\n\n"
        f"Parameters:\n"
        f"- Target number of shots: {num_shots}\n"
        f"- Duration per shot: {shot_duration}s (Total: {total_duration}s)\n"
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
        scenes: list[StoryboardScene] = []
        time_per_shot = round(total_duration / max(1, len(raw_shots) or num_shots), 2)

        for i, s_raw in enumerate(raw_shots[:num_shots]):
            idx = i + 1
            ts_str = str(s_raw.get("timestamp_str") or format_timestamp(i * time_per_shot))
            desc = str(s_raw.get("description") or f"Scene {idx}")
            scenes.append(
                StoryboardScene(
                    index=idx,
                    title=f"Plan {idx}",
                    start_time=i * time_per_shot,
                    duration=time_per_shot,
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
        log.warning("LLM Storyboard call failed (%s); falling back to Built-in Engine: %s", endpoint, exc)
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
    total_frames = sum(max(5, int(round(s.duration * fps))) for s in scenes)

    # Reference pictures from subjects or scenes
    ref_pictures: list[dict[str, Any]] = []
    if subjects:
        for subj in subjects:
            if subj.image_b64:
                ref_pictures.append({
                    "index": subj.index - 1,
                    "imageB64": subj.image_b64,
                    "assetId": f"subject-{subj.index}",
                })
    elif scenes:
        for s in scenes:
            if s.image_b64:
                ref_pictures.append({
                    "index": s.index - 1,
                    "imageB64": s.image_b64,
                    "assetId": f"scene-{s.index}",
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
                    "pictures": ref_pictures,
                    "identityPictures": ref_pictures,
                    "startFrame": {
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
                "refs": ref_pictures,
            },
            "segments": mixed_segments,
        }
        return timeline_dict

    # Standard / Global timeline structure
    cursor = 0
    std_segments = []
    task_key = "r2v" if (director_mode in ("r2v", "mixed") or ref_pictures) else director_mode
    for s in scenes:
        seg_frames = max(5, int(round(s.duration * fps)))
        std_segments.append(
            {
                "id": f"s{s.index - 1}",
                "start": cursor,
                "length": seg_frames,
                "prompt": s.full_prompt,
                "taskType": task_key,
                "refs": ref_pictures,
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
        "global": {
            "taskType": task_key,
            "prompt": effective_prompt,
            "refs": ref_pictures,
        },
        "r2vCommon": {
            "refs": ref_pictures,
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
