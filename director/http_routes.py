# Portions derived from ComfyUI_MiniMaxH3_Director
# Copyright AIMixer and contributors
# Originally licensed under Apache License 2.0
# Modified for MiniMax H3 Motion Director, 2026-08-09
# This derivative project is distributed under GPL-3.0.
# See NOTICE and LICENSES/Apache-2.0-AIMixer.txt.

"""HTTP routes for MiniMax H3 Motion Director."""

from __future__ import annotations

import logging
import os
import re
import shutil
import importlib.util

import folder_paths
from aiohttp import web
from server import PromptServer

from .face_refine_validation import compatible_sam_models
from .material_library_routes import register_material_library_routes
from .video_export import (
    FINAL_VIDEO_REGISTRY,
    FinalVideoUnavailable,
    StaleFinalVideoRun,
    video_save_capabilities,
)

log = logging.getLogger("ComfyUI-MiniMax-H3-Motion-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\-()\u4e00-\u9fff]+")
_ROUTES_REGISTERED = False


def _safe_basename(name: str) -> str:
    base = os.path.basename(str(name or "video.mp4").replace("\\", "/"))
    base = _SAFE_NAME.sub("_", base).strip("._")
    return base or "video.mp4"


async def minimax_upload_video_chunk(request):
    try:
        post = await request.post()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid upload: {exc}")

    upload_id = str(post.get("upload_id") or "").strip()
    filename = _safe_basename(post.get("filename"))
    chunk_field = post.get("chunk")
    if not upload_id or chunk_field is None:
        return web.Response(status=400, text="Missing upload_id or chunk.")

    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return web.Response(status=400, text="Invalid upload_id.")

    try:
        chunk_index = int(post.get("chunk_index", 0))
        total_chunks = int(post.get("total_chunks", 1))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid chunk index.")

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        return web.Response(status=400, text="Chunk index out of range.")

    session_dir = os.path.join(CHUNK_ROOT, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")

    with open(part_path, "wb") as out:
        while True:
            block = chunk_field.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)

    if chunk_index + 1 < total_chunks:
        return web.json_response({"status": "ok", "chunk_index": chunk_index})

    input_dir = folder_paths.get_input_directory()
    out_path = os.path.join(input_dir, filename)
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        for n in range(1, 1000):
            candidate = f"{stem}_{n}{ext}"
            candidate_path = os.path.join(input_dir, candidate)
            if not os.path.exists(candidate_path):
                out_path = candidate_path
                filename = candidate
                break

    with open(out_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(session_dir, f"{i:06d}.part")
            if not os.path.isfile(part):
                shutil.rmtree(session_dir, ignore_errors=True)
                return web.Response(status=400, text=f"Missing chunk {i}.")
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(session_dir, ignore_errors=True)
    log.info("MiniMax H3 Motion Director uploaded video to input/: %s", filename)
    return web.json_response({"name": filename, "subfolder": "", "type": "input"})


async def minimax_probe_video(request):
    try:
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        else:
            body = dict(request.query)
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid request: {exc}")

    video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
    if not video_file:
        return web.Response(status=400, text="Missing videoFile.")

    from ..lib.video_io import probe_video_clip

    clip = {
        "videoFile": video_file,
        "fileName": os.path.basename(video_file),
        "subfolder": str(body.get("subfolder") or "").strip(),
        "type": str(body.get("type") or "input").strip() or "input",
    }
    try:
        info = probe_video_clip(clip)
    except Exception as exc:
        log.warning("MiniMax H3 Motion Director video probe failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    return web.json_response(info)


async def minimax_detect_shots(request):
    """Detect shot boundaries with PySceneDetect; return logical cut frames."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    from ..lib.shot_detect import (
        detect_timeline_shot_cuts,
        scenedetect_available,
        scenedetect_install_hint,
    )

    if not scenedetect_available():
        return web.Response(
            status=400,
            text=(
                "PySceneDetect is not installed in ComfyUI's Python "
                f"({__import__('sys').executable}). "
                f"Run: {scenedetect_install_hint()}"
            ),
        )

    try:
        frame_rate = float(body.get("frameRate") or body.get("frame_rate") or 24)
    except (TypeError, ValueError):
        frame_rate = 24.0
    try:
        total_frames = int(body.get("totalFrames") or body.get("total_frames") or 0)
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid totalFrames.")

    sensitivity = str(body.get("sensitivity") or "medium").strip().lower()
    try:
        min_shot_frames = int(body.get("minShotFrames") or body.get("min_shot_frames") or 12)
    except (TypeError, ValueError):
        min_shot_frames = 12

    clips_in = body.get("clips")
    clips: list[dict] = []
    if isinstance(clips_in, list) and clips_in:
        for item in clips_in:
            if not isinstance(item, dict):
                continue
            video_file = str(item.get("videoFile") or item.get("video_file") or "").strip()
            if not video_file:
                continue
            clips.append(
                {
                    "videoFile": video_file,
                    "fileName": os.path.basename(video_file),
                    "subfolder": str(item.get("subfolder") or "").strip(),
                    "type": str(item.get("type") or "input").strip() or "input",
                    "logicalStart": item.get("logicalStart", item.get("logical_start", 0)),
                    "logicalEnd": item.get("logicalEnd", item.get("logical_end", total_frames)),
                    "nativeFps": item.get("nativeFps", item.get("native_fps")),
                }
            )
    else:
        video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing clips[] or videoFile.")
        clips.append(
            {
                "videoFile": video_file,
                "fileName": os.path.basename(video_file),
                "subfolder": str(body.get("subfolder") or "").strip(),
                "type": str(body.get("type") or "input").strip() or "input",
                "logicalStart": 0,
                "logicalEnd": total_frames,
                "nativeFps": body.get("nativeFps", body.get("native_fps")),
            }
        )

    if total_frames <= 0:
        return web.Response(status=400, text="totalFrames must be > 0.")

    try:
        result = detect_timeline_shot_cuts(
            clips,
            frame_rate=frame_rate,
            total_frames=total_frames,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
        )
    except ImportError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Motion Director shot detect failed: %s", exc)
        return web.Response(status=400, text=str(exc))

    return web.json_response(result)


def _register_route(routes, method: str, path: str, handler) -> None:
    if hasattr(routes, "add_route"):
        routes.add_route(method, path, handler)
    elif method == "POST" and hasattr(routes, "post"):
        routes.post(path)(handler)
    elif method == "GET" and hasattr(routes, "get"):
        routes.get(path)(handler)
    else:
        raise AttributeError("Unsupported ComfyUI route table API")


async def minimax_postprocess_capabilities(_request):
    """Lazy dependency/model discovery for modal selectors only."""
    def filenames(*categories):
        result = []
        for category in categories:
            try:
                result.extend(folder_paths.get_filename_list(category) or [])
            except Exception:
                continue
        return sorted(dict.fromkeys(str(value) for value in result))

    return web.json_response({
        "diffusion_models": filenames("diffusion_models"),
        "upscale_models": filenames("upscale_models"),
        "latent_upscale_models": filenames("latent_upscale_models"),
        "face_detectors": filenames("ultralytics_bbox", "ultralytics"),
        "sam_models": compatible_sam_models(),
        "sam_model_folder": os.path.join(folder_paths.models_dir, "sams"),
        "dependencies": {
            "nvidia_rtx_vsr": importlib.util.find_spec("nvvfx") is not None,
            "ultralytics": importlib.util.find_spec("ultralytics") is not None,
            "insightface": importlib.util.find_spec("insightface") is not None,
            "sam": importlib.util.find_spec("ultralytics") is not None,
        },
        "video_save": video_save_capabilities(),
    })


async def minimax_save_final_video(request):
    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
    node_id = str(body.get("node_id") or "").strip()
    run_id = str(body.get("run_id") or "").strip()
    if not node_id or not run_id:
        return web.json_response({"ok": False, "error": "Missing node_id or run_id"}, status=400)
    try:
        result = FINAL_VIDEO_REGISTRY.save(
            node_id,
            run_id,
            body.get("save") or body,
            segment_range=body.get("segment_range"),
        )
        return web.json_response(result)
    except StaleFinalVideoRun as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=409)
    except FinalVideoUnavailable as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=404)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        log.exception("MiniMax H3 Motion Director Final Result save failed")
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


async def minimax_release_final_video(request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    node_id = str(body.get("node_id") or "").strip()
    if node_id:
        FINAL_VIDEO_REGISTRY.release(node_id)
    return web.json_response({"ok": True})


async def minimax_storyboard_generate(request):
    """Generate storyboard and sync directly to Motion Director without queuing full workflow render."""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        import json
        from .storyboard import (
            DEFAULT_NEGATIVE_PROMPT,
            DEFAULT_STYLE_PROMPT,
            build_director_timeline,
            generate_llm_storyboard,
            generate_local_storyboard,
            generate_subject_reference_images,
            parse_aspect_ratio,
        )

        story = str(body.get("story") or "").strip()
        total_duration_sec = float(body.get("total_duration_sec") or 15.0)
        shot_duration_sec = float(body.get("shot_duration_sec") or 3.0)
        aspect_ratio = str(body.get("aspect_ratio") or "16:9 (864x480)")
        director_mode = str(body.get("director_mode") or "mixed")
        style_prompt = str(body.get("style_prompt") or DEFAULT_STYLE_PROMPT)
        negative_prompt = str(body.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)
        ai_story_engine = str(body.get("ai_story_engine") or "Built-in AI Story Engine (Local / No Server needed)")
        llm_url = str(body.get("llm_url") or "http://127.0.0.1:11434")
        llm_model = str(body.get("llm_model") or "qwen2.5:7b")
        image_steps = int(body.get("image_steps") or 8)
        image_cfg = float(body.get("image_cfg") or 2.0)
        seed = int(body.get("seed") or 0)
        node_id = str(body.get("node_id") or "").strip()

        width, height = parse_aspect_ratio(aspect_ratio)

        sb_data = None
        engine_lower = ai_story_engine.lower()
        if "ollama" in engine_lower:
            sb_data = generate_llm_storyboard(
                story=story,
                total_duration=total_duration_sec,
                shot_duration=shot_duration_sec,
                style_prompt=style_prompt,
                llm_url=llm_url,
                llm_model=llm_model,
                api_format="ollama",
            )
        elif "openai" in engine_lower:
            sb_data = generate_llm_storyboard(
                story=story,
                total_duration=total_duration_sec,
                shot_duration=shot_duration_sec,
                style_prompt=style_prompt,
                llm_url=llm_url,
                llm_model=llm_model,
                api_format="openai",
            )

        if not sb_data:
            sb_data = generate_local_storyboard(
                story=story,
                total_duration=total_duration_sec,
                shot_duration=shot_duration_sec,
                style_prompt=style_prompt,
            )

        # Generate subject references
        batch_images, b64_list = generate_subject_reference_images(
            sb_data.subjects,
            width=width,
            height=height,
            model=None,
            clip=None,
            vae=None,
            steps=image_steps,
            cfg=image_cfg,
            seed=seed,
            negative_prompt=negative_prompt,
            external_images=None,
        )

        # Build timeline dict
        timeline_dict = build_director_timeline(
            sb_data.scenes,
            subjects=sb_data.subjects,
            full_h3_prompt=sb_data.full_h3_prompt,
            director_mode=director_mode,
            width=width,
            height=height,
            fps=24.0,
            ref_max_size=max(width, height),
        )
        timeline_json = json.dumps(timeline_dict, ensure_ascii=False, indent=2)

        # Broadcast event to connected nodes in ComfyUI frontend
        try:
            from server import PromptServer
            if PromptServer.instance:
                PromptServer.instance.send_sync(
                    "minimax_storyboard_generated",
                    {
                        "node_id": node_id,
                        "timeline_data": timeline_json,
                        "total_duration": total_duration_sec,
                        "director_mode": director_mode,
                        "num_scenes": len(sb_data.scenes),
                        "summary": sb_data.full_h3_prompt,
                        "prompt": sb_data.full_h3_prompt,
                        "subjects": [s.to_dict() for s in sb_data.subjects],
                        "reference_images": [s.image_b64 for s in sb_data.subjects],
                    },
                )
        except Exception as exc:
            log.warning("PromptServer broadcast error: %s", exc)

        return web.json_response({
            "ok": True,
            "timeline_data": timeline_json,
            "timeline_dict": timeline_dict,
            "prompt": sb_data.full_h3_prompt,
            "summary": sb_data.full_h3_prompt,
            "director_mode": director_mode,
            "total_duration": total_duration_sec,
            "subjects": [s.to_dict() for s in sb_data.subjects],
            "scenes": [s.to_dict() for s in sb_data.scenes],
            "reference_images": [s.image_b64 for s in sb_data.subjects],
        })
    except Exception as exc:
        log.error("Storyboard generation failed: %s", exc, exc_info=True)
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


def register_routes() -> bool:
    """Register MiniMax H3 Motion Director HTTP routes on the ComfyUI PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    server = PromptServer.instance
    if server is None:
        log.warning("MiniMax H3 Motion Director: PromptServer not ready, HTTP routes not registered")
        return False

    routes = server.routes
    _register_route(routes, "POST", "/minimax/motion-director/upload_chunk", minimax_upload_video_chunk)
    _register_route(routes, "POST", "/minimax/motion-director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/motion-director/probe_video", minimax_probe_video)
    _register_route(routes, "POST", "/minimax/motion-director/detect_shots", minimax_detect_shots)
    _register_route(routes, "GET", "/minimax/motion-director/postprocess_capabilities", minimax_postprocess_capabilities)
    _register_route(routes, "POST", "/minimax/motion-director/save_video", minimax_save_final_video)
    _register_route(routes, "POST", "/minimax/motion-director/release_video", minimax_release_final_video)
    _register_route(routes, "POST", "/minimax/storyboard/generate", minimax_storyboard_generate)
    register_material_library_routes(routes)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Motion Director HTTP routes registered")
    return True
