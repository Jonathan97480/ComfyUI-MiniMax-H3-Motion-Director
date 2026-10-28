"""Per-segment R2V Common/Local reference compilation.

The timeline stores stable semantic asset IDs.  Official MiniMax tags are a
runtime presentation detail and are rebuilt for every segment from its actual
selected Common assets followed by its Local assets.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.effective_refs")

SEMANTIC_TOKEN_RE = re.compile(
    r"\{\{mmx-ref:(picture|video|audio):([A-Za-z0-9_.:-]+)\}\}",
    re.IGNORECASE,
)


class SemanticReferenceError(ValueError):
    """A semantic prompt reference cannot be resolved for this segment."""


@dataclass(frozen=True)
class EffectiveReferences:
    pictures: list[Any]
    videos: list[Any]
    audios: list[Any]
    video_audios: list[Any]
    tags: dict[tuple[str, str], str]


def semantic_reference_token(kind: str, asset_id: str) -> str:
    key = str(kind or "").strip().lower()
    identity = str(asset_id or "").strip()
    if key not in {"picture", "video", "audio"}:
        raise ValueError("Semantic reference kind must be picture, video, or audio.")
    if not identity or not re.fullmatch(r"[A-Za-z0-9_.:-]+", identity):
        raise ValueError("Semantic reference asset ID is empty or contains unsafe characters.")
    return f"{{{{mmx-ref:{key}:{identity}}}}}"


def _asset_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("assetId") or item.get("asset_id") or item.get("id") or "").strip()
    return str(getattr(item, "asset_id", "") or "").strip()


def _item_index(item: Any) -> int:
    if isinstance(item, dict):
        return int(item.get("index", item.get("slot", 0)))
    return int(getattr(item, "index", 0))


def _is_same_ref(a: Any, b: Any) -> bool:
    if a is b:
        return True
    id_a = _asset_id(a)
    id_b = _asset_id(b)
    if id_a and id_b and id_a == id_b:
        return True
    tensor_a = getattr(a, "tensor", None) if not isinstance(a, dict) else a.get("tensor")
    tensor_b = getattr(b, "tensor", None) if not isinstance(b, dict) else b.get("tensor")
    if tensor_a is not None and tensor_b is not None:
        if tensor_a is tensor_b:
            return True
        try:
            if hasattr(tensor_a, "data_ptr") and hasattr(tensor_b, "data_ptr"):
                if tensor_a.data_ptr() != 0 and tensor_a.data_ptr() == tensor_b.data_ptr():
                    return True
            if hasattr(tensor_a, "shape") and hasattr(tensor_b, "shape"):
                if tensor_a.shape == tensor_b.shape:
                    import torch
                    if torch.equal(tensor_a, tensor_b):
                        return True
        except Exception:
            pass
    file_a = (
        (getattr(a, "image_file", None) or getattr(a, "video_file", None) or getattr(a, "audio_file", None))
        if not isinstance(a, dict)
        else (a.get("imageFile") or a.get("videoFile") or a.get("audioFile"))
    )
    file_b = (
        (getattr(b, "image_file", None) or getattr(b, "video_file", None) or getattr(b, "audio_file", None))
        if not isinstance(b, dict)
        else (b.get("imageFile") or b.get("videoFile") or b.get("audioFile"))
    )
    if file_a and file_b and file_a == file_b:
        return True
    b64_a = getattr(a, "image_b64", None) if not isinstance(a, dict) else a.get("imageB64")
    b64_b = getattr(b, "image_b64", None) if not isinstance(b, dict) else b.get("imageB64")
    if b64_a and b64_b and b64_a == b64_b:
        return True
    return False


def _renumber(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for index, item in enumerate(items):
        cloned = copy.copy(item)
        if isinstance(cloned, dict):
            cloned["index"] = index
            cloned["slot"] = index
        else:
            cloned.index = index
        out.append(cloned)
    return out


def _selected(items: Iterable[Any], selected_ids: set[str]) -> list[Any]:
    return [item for item in items if _asset_id(item) in selected_ids]


def _paired_for_videos(video_audios: Iterable[Any], videos: list[Any]) -> list[Any]:
    """Return paired soundtracks in effective video order.

    A paired audio shares the stable asset ID of its Video asset.  ``index`` is
    rewritten to the Video input index because the official node looks up
    ``ref_video_audio_N`` by ``ref_video_N``.
    """
    by_asset = {_asset_id(item): item for item in video_audios if _asset_id(item)}
    out: list[Any] = []
    for video in videos:
        asset_id = _asset_id(video)
        paired = by_asset.get(asset_id)
        if paired is None:
            continue
        cloned = copy.copy(paired)
        v_idx = _item_index(video)
        if isinstance(cloned, dict):
            cloned["index"] = v_idx
            cloned["slot"] = v_idx
        else:
            cloned.index = v_idx
        out.append(cloned)
    return out


def compile_effective_references(
    *,
    common_pictures: Iterable[Any] = (),
    common_videos: Iterable[Any] = (),
    common_audios: Iterable[Any] = (),
    common_video_audios: Iterable[Any] = (),
    selected_common_asset_ids: set[str] | frozenset[str] = frozenset(),
    local_pictures: Iterable[Any] = (),
    local_videos: Iterable[Any] = (),
    local_audios: Iterable[Any] = (),
    local_video_audios: Iterable[Any] = (),
) -> EffectiveReferences:
    """Build effective references and exact official prompt-tag mapping.

    Ordering is deterministic: selected Common assets in pool order, then Local
    assets in segment order.  Picture/Video slots are independently dense.
    Official Audio numbering is paired Video soundtracks first (Video order),
    then standalone Audio assets, matching ``MiniMaxH3ReferenceToVideo``.
    Enforces hardware/API limits defensively (max 9 pictures, 3 videos, 3 audios).
    """
    selected_ids = {str(value) for value in selected_common_asset_ids}
    alias_map: dict[tuple[str, str], str] = {}

    # 1. Pictures (MiniMax H3 allows at most 9 pictures)
    sel_common_pics = _selected(common_pictures, selected_ids)
    loc_pics = list(local_pictures)
    dedup_common_pics: list[Any] = []
    for c in sel_common_pics:
        matched = next((loc for loc in loc_pics if _is_same_ref(c, loc)), None)
        if matched is not None:
            c_id = _asset_id(c)
            l_id = _asset_id(matched)
            if c_id and l_id:
                alias_map[("picture", c_id)] = l_id
        else:
            dedup_common_pics.append(c)

    if len(loc_pics) >= 9:
        if len(loc_pics) > 9:
            log.warning(
                "Segment local pictures exceed MiniMax H3 limit of 9 (%d provided); clamped to 9.",
                len(loc_pics),
            )
        chosen_pics = loc_pics[:9]
    else:
        max_common = 9 - len(loc_pics)
        if len(dedup_common_pics) > max_common:
            log.warning(
                "Segment effective pictures (%d common + %d local) exceed limit of 9; clamped to 9.",
                len(dedup_common_pics),
                len(loc_pics),
            )
        chosen_pics = [*dedup_common_pics[:max_common], *loc_pics]
    pictures = _renumber(chosen_pics)

    # 2. Videos (MiniMax H3 allows at most 3 videos)
    sel_common_videos = _selected(common_videos, selected_ids)
    loc_videos = list(local_videos)
    dedup_common_videos: list[Any] = []
    for c in sel_common_videos:
        matched = next((loc for loc in loc_videos if _is_same_ref(c, loc)), None)
        if matched is not None:
            c_id = _asset_id(c)
            l_id = _asset_id(matched)
            if c_id and l_id:
                alias_map[("video", c_id)] = l_id
        else:
            dedup_common_videos.append(c)

    if len(loc_videos) >= 3:
        if len(loc_videos) > 3:
            log.warning(
                "Segment local videos exceed MiniMax H3 limit of 3 (%d provided); clamped to 3.",
                len(loc_videos),
            )
        chosen_videos = loc_videos[:3]
    else:
        max_common_vids = 3 - len(loc_videos)
        chosen_videos = [*dedup_common_videos[:max_common_vids], *loc_videos]
    videos = _renumber(chosen_videos)

    # 3. Standalone Audios (MiniMax H3 allows at most 3 standalone audios)
    sel_common_audios = _selected(common_audios, selected_ids)
    loc_audios = list(local_audios)
    dedup_common_audios: list[Any] = []
    for c in sel_common_audios:
        matched = next((loc for loc in loc_audios if _is_same_ref(c, loc)), None)
        if matched is not None:
            c_id = _asset_id(c)
            l_id = _asset_id(matched)
            if c_id and l_id:
                alias_map[("audio", c_id)] = l_id
        else:
            dedup_common_audios.append(c)

    if len(loc_audios) >= 3:
        if len(loc_audios) > 3:
            log.warning(
                "Segment local audios exceed MiniMax H3 limit of 3 (%d provided); clamped to 3.",
                len(loc_audios),
            )
        chosen_audios = loc_audios[:3]
    else:
        max_common_auds = 3 - len(loc_audios)
        chosen_audios = [*dedup_common_audios[:max_common_auds], *loc_audios]
    audios = _renumber(chosen_audios)

    selected_common_video_audios = _selected(common_video_audios, selected_ids)
    video_audios = _paired_for_videos(
        [*selected_common_video_audios, *list(local_video_audios)], videos
    )

    tags: dict[tuple[str, str], str] = {}
    for item in pictures:
        if _asset_id(item):
            tags[("picture", _asset_id(item))] = f"<Picture {_item_index(item) + 1}>"
    for item in videos:
        if _asset_id(item):
            tags[("video", _asset_id(item))] = f"<Video {_item_index(item) + 1}>"

    audio_number = 1
    for item in video_audios:
        if _asset_id(item):
            tags[("audio", _asset_id(item))] = f"<Audio {audio_number}>"
        audio_number += 1
    for item in audios:
        if _asset_id(item):
            tags[("audio", _asset_id(item))] = f"<Audio {audio_number}>"
        audio_number += 1

    # Map aliases so semantic tags for deduplicated items resolve to the active tag
    for (kind, dup_id), can_id in alias_map.items():
        if (kind, can_id) in tags and (kind, dup_id) not in tags:
            tags[(kind, dup_id)] = tags[(kind, can_id)]

    return EffectiveReferences(
        pictures=pictures,
        videos=videos,
        audios=audios,
        video_audios=video_audios,
        tags=tags,
    )


def compile_semantic_prompt(
    prompt: str | None,
    tags: Mapping[tuple[str, str], str],
    *,
    known_assets: Mapping[tuple[str, str], str] | None = None,
    segment_label: str = "Segment",
) -> str:
    """Replace semantic tokens with official tags or fail on disabled assets."""
    text = str(prompt or "")
    known = known_assets or {}

    def replace(match: re.Match[str]) -> str:
        kind = match.group(1).lower()
        asset_id = match.group(2)
        key = (kind, asset_id)
        official = tags.get(key)
        if official is not None:
            return official
        label = known.get(key)
        pretty_kind = {"picture": "Picture", "video": "Video", "audio": "Audio"}[kind]
        if label:
            raise SemanticReferenceError(
                f'{segment_label} prompt references {label} ({pretty_kind} asset "{asset_id}"), '
                "but that asset is disabled for this segment. Re-enable it or remove "
                "the corresponding prompt reference."
            )
        raise SemanticReferenceError(
            f'{segment_label} prompt references unknown {pretty_kind} asset "{asset_id}". '
            "Restore the asset or remove the corresponding prompt reference."
        )

    return SEMANTIC_TOKEN_RE.sub(replace, text)


__all__ = [
    "EffectiveReferences",
    "SEMANTIC_TOKEN_RE",
    "SemanticReferenceError",
    "compile_effective_references",
    "compile_semantic_prompt",
    "semantic_reference_token",
]
