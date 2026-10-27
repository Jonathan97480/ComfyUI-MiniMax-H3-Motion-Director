# Tests for MiniMax H3 AI Story Director
# Author: Jonathan97480 & Contributors
# Licensed under GNU GPL v3.0

import json
import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from director.storyboard import (
    ASPECT_RATIOS,
    MMX_STORYBOARD,
    StoryboardData,
    StoryboardScene,
    StoryboardSubject,
    build_director_timeline,
    format_script_summary,
    generate_local_scenes,
    generate_local_storyboard,
    generate_scene_reference_images,
    generate_subject_reference_images,
    parse_aspect_ratio,
    parse_existing_h3_prompt,
    render_storyboard_visual_concept,
)
from nodes.storyboard_node import MiniMaxH3StoryDirector


USER_EXAMPLE_PROMPT = """subject_definitions:

<Subject 1> Girl, based on <Picture 1>: teenage anime girl whose facial identity, hairstyle, age appearance, body proportions, clothing and overall character design are locked to the supplied character reference. Preserve her recognizable appearance throughout every camera cut. The background of <Picture 1> is not used.

<Subject 2> Bedroom, based on <Picture 2>: the exact cozy teenage bedroom environment shown in the supplied reference, including the bed, window, curtains, desk, furniture arrangement, materials, warm neutral colors and spatial layout. The layout is fixed: the window and desk are on the left side of the frame, and the bed is on the right side of the frame. Preserve this environment and layout consistently throughout the entire video. The character shown in <Picture 2>, if any, is ignored; <Picture 1> controls the girl's identity.

summary:

[reference generation] A quiet morning in <Subject 2>. <Subject 1> wakes as the room brightens with sunlight, gets up, takes her headphones from the desk and walks out of frame. The sequence uses deliberate cinematic cuts between meaningful viewpoints while preserving the same character and bedroom.

retention_analysis:

<Subject 1> (appears throughout): fully_preserved — preserve the exact facial identity, hairstyle, age appearance, body proportions, clothing and recognizable character design from <Picture 1>. Allow natural changes in pose, expression and movement.

<Subject 2> (appears throughout): fully_preserved — preserve the exact bedroom architecture, bed position on the right, window and desk position on the left, curtains, furniture, materials, lighting direction and spatial relationships from <Picture 2>.

detailed_description:

The target video is a polished cinematic slice-of-life anime sequence set during a bright morning. Movement is natural, understated and physically believable. The bedroom layout is fixed throughout: the window and desk remain on the left side of the frame, and the bed remains on the right side of the frame. H3 should use real cinematic cuts between clearly different viewpoints; each cut introduces meaningful new framing or visual information.

[Shot 1] Begin with a wide static establishing view of <Subject 2>, matching the layout in <Picture 2>: the window and desk on the left side of the frame, the bed on the right side. The bedroom is quiet and dim in the early morning. <Subject 1> lies asleep on the bed on the right side of the frame with minimal natural movement. Soft morning light gradually brightens the room through the window on the left, and the curtains move gently in the breeze. As the light grows warmer and brighter across the bed, <Subject 1> stirs, shifts, and slowly opens her eyes. Hold this establishing composition briefly so the viewer understands the room layout.

At approximately 00:03.000, make a hard cinematic cut to an extreme close-up of <Subject 1>'s eyes. Her eyes are open and adjusting to the light, and she blinks naturally as she wakes.

At approximately 00:04.500, make a hard cut to a medium shot that includes both <Subject 1> on the bed on the right side of the frame and the window on the left side of the frame. She turns her head to the left and gazes out through the window. Her expression becomes quietly interested rather than dramatically surprised.

At approximately 00:06.500, make a hard cut to a close three-quarter view of the desk area on the left side of the room. The headphones are clearly visible resting on the desk. <Subject 1> walks to the desk and grabs the headphones.

At approximately 00:08.500, continue in the same close composition as <Subject 1> picks up the headphones with one hand. Her movement is deliberate and natural. She turns away from the desk and stands upright, facing the open room.

At approximately 00:10.000, make a hard cut to a rear medium shot from inside the bedroom, framing <Subject 1> from behind as she walks a few steps across the room, moving away from the desk toward the right side of the frame with the headphones in her hand. The camera remains static as she continues walking and passes out of frame on the right.

End with the bedroom visible, the curtains still moving gently in the breeze, and no doorway shown.

overall_soundscape:

Quiet intimate bedroom ambience throughout. Soft morning wind enters through the window and gently moves the curtains. Very subtle curtain and fabric movement. Natural mattress and clothing sounds as <Subject 1> stirs and wakes. Soft footsteps across the bedroom floor as she approaches the desk and walks across the room. A subtle natural handling sound occurs when she picks up the headphones. Distant small birds can be heard outside the window throughout, with occasional gentle bird calls.

non_diegetic_music:

N/A"""


class TestStoryboardModule(unittest.TestCase):
    def test_parse_aspect_ratio(self):
        w, h = parse_aspect_ratio("16:9 (864x480)")
        self.assertEqual(w, 864)
        self.assertEqual(h, 480)

        w, h = parse_aspect_ratio("1:1 (768x768)")
        self.assertEqual(w, 768)
        self.assertEqual(h, 768)

        w, h = parse_aspect_ratio("custom 1024x512")
        self.assertEqual(w, 1024)
        self.assertEqual(h, 512)

    def test_generate_local_storyboard(self):
        story = (
            "Un chevalier cyberpunk explore les ruelles d'une ville neon sous la pluie battante. "
            "Il rencontre un renard holographique lumineux qui le guide vers une porte secrete dans un temple antique futuriste."
        )
        sb_data = generate_local_storyboard(
            story=story,
            total_duration=15.0,
            shot_duration=3.0,
            style_prompt="cinematic, 8k",
        )
        self.assertIsInstance(sb_data, StoryboardData)
        self.assertGreaterEqual(len(sb_data.subjects), 2)
        # Check Subject 1 is character, Subject 2 is environment
        self.assertEqual(sb_data.subjects[0].category, "character")
        self.assertEqual(sb_data.subjects[1].category, "environment")
        self.assertIn("subject_definitions:", sb_data.full_h3_prompt)
        self.assertIn("<Subject 1>", sb_data.full_h3_prompt)
        self.assertIn("<Subject 2>", sb_data.full_h3_prompt)
        self.assertIn("<Picture 1>", sb_data.full_h3_prompt)
        self.assertIn("<Picture 2>", sb_data.full_h3_prompt)
        self.assertIn("detailed_description:", sb_data.full_h3_prompt)
        self.assertIn("overall_soundscape:", sb_data.full_h3_prompt)
        self.assertGreaterEqual(len(sb_data.scenes), 3)

    def test_parse_existing_h3_prompt(self):
        sb_data = parse_existing_h3_prompt(USER_EXAMPLE_PROMPT, total_duration=12.0, style_prompt="8k, masterpiece")
        self.assertIsNotNone(sb_data)
        self.assertEqual(len(sb_data.subjects), 2)
        s1 = sb_data.subjects[0]
        s2 = sb_data.subjects[1]
        self.assertEqual(s1.name, "Girl")
        self.assertEqual(s1.category, "character")
        self.assertEqual(s2.name, "Bedroom")
        self.assertEqual(s2.category, "environment")
        # Check that image prompts were cleaned of video constraints
        self.assertNotIn("The background of <Picture 1> is not used", s1.image_prompt)
        self.assertNotIn("Preserve this environment", s2.image_prompt)
        # Check shots with timestamps
        self.assertEqual(len(sb_data.scenes), 6)
        self.assertEqual(sb_data.scenes[0].timestamp_str, "00:00.000")
        self.assertEqual(sb_data.scenes[1].timestamp_str, "00:03.000")
        self.assertEqual(sb_data.scenes[2].timestamp_str, "00:04.500")

    def test_render_storyboard_visual_concept(self):
        tensor = render_storyboard_visual_concept(
            scene_num=1,
            title="Girl",
            story_snippet="teenage anime girl",
            lighting="Soft diffused morning light",
            category="character",
            width=864,
            height=480,
            seed=42,
        )
        self.assertEqual(tensor.shape, (1, 480, 864, 3))
        self.assertTrue(torch.is_floating_point(tensor))
        self.assertTrue(tensor.min() >= 0.0)
        self.assertTrue(tensor.max() <= 1.0)

    def test_build_director_timeline(self):
        sb_data = generate_local_storyboard(
            story="Test story in a bedroom",
            total_duration=10.0,
            shot_duration=5.0,
            style_prompt="photorealistic",
        )
        sb_data.subjects[0].image_b64 = "data:image/jpeg;base64,subject1"
        sb_data.subjects[1].image_b64 = "data:image/jpeg;base64,subject2"

        timeline = build_director_timeline(
            sb_data.scenes,
            subjects=sb_data.subjects,
            full_h3_prompt=sb_data.full_h3_prompt,
            director_mode="mixed",
            width=864,
            height=480,
            fps=24.0,
        )
        self.assertEqual(timeline["timelineMode"], "mixed")
        self.assertGreaterEqual(len(timeline["segments"]), 2)
        # All segments must share the same canonical picture references!
        seg0_pics = timeline["segments"][0]["inputs"]["pictures"]
        seg1_pics = timeline["segments"][1]["inputs"]["pictures"]
        self.assertEqual(len(seg0_pics), 2)
        self.assertEqual(len(seg1_pics), 2)
        self.assertEqual(seg0_pics[0]["imageB64"], "data:image/jpeg;base64,subject1")
        self.assertEqual(seg0_pics[1]["imageB64"], "data:image/jpeg;base64,subject2")
        self.assertEqual(seg1_pics[0]["imageB64"], "data:image/jpeg;base64,subject1")
        self.assertEqual(seg1_pics[1]["imageB64"], "data:image/jpeg;base64,subject2")

    def test_node_execution_free_form(self):
        node = MiniMaxH3StoryDirector()
        sb, t_json, imgs, summary = node.execute(
            story="Le samouraï dégaine son katana énergétique sous la pluie devant un temple antique.",
            total_duration_sec=10.0,
            shot_duration_sec=5.0,
            aspect_ratio="16:9 (864x480)",
            director_mode="mixed",
            style_prompt="8k, masterpiece",
            negative_prompt="blurry, text",
            ai_story_engine="Built-in AI Story Engine (Local / No Server needed)",
            llm_url="http://127.0.0.1:11434",
            llm_model="qwen2.5:7b",
            image_steps=8,
            image_cfg=2.0,
            seed=123,
            control_after_generate="fixed",
        )
        self.assertIsInstance(sb, dict)
        self.assertEqual(sb["version"], 1)
        # Canonical reference images batch has shape (N, H, W, 3) for Subject 1, Subject 2, and optional Subject 3
        self.assertGreaterEqual(imgs.shape[0], 2)
        self.assertEqual(imgs.shape[1:], (480, 864, 3))
        self.assertIn("timelineMode", t_json)
        self.assertIn("subject_definitions:", summary)
        self.assertIn("<Subject 1>", summary)
        self.assertIn("<Picture 1>", summary)

    def test_node_execution_preformatted_prompt(self):
        node = MiniMaxH3StoryDirector()
        sb, t_json, imgs, summary = node.execute(
            story=USER_EXAMPLE_PROMPT,
            total_duration_sec=12.0,
            shot_duration_sec=2.0,
            aspect_ratio="16:9 (864x480)",
            director_mode="mixed",
            style_prompt="8k, masterpiece",
            negative_prompt="blurry, text",
            ai_story_engine="Built-in AI Story Engine (Local / No Server needed)",
            llm_url="http://127.0.0.1:11434",
            llm_model="qwen2.5:7b",
            image_steps=8,
            image_cfg=2.0,
            seed=123,
            control_after_generate="fixed",
        )
        self.assertIsInstance(sb, dict)
        self.assertEqual(imgs.shape, (2, 480, 864, 3))
        self.assertIn("<Subject 1> Girl, based on <Picture 1>", summary)
        self.assertIn("<Subject 2> Bedroom, based on <Picture 2>", summary)

    def test_defensive_coercion(self):
        node = MiniMaxH3StoryDirector()
        sb, t_json, imgs, summary = node.execute(
            story="Test story",
            total_duration_sec="10.0",
            shot_duration_sec="5.0",
            aspect_ratio="16:9 (864x480)",
            director_mode="mixed",
            style_prompt="8k",
            negative_prompt="blurry",
            ai_story_engine="Built-in AI Story Engine (Local / No Server needed)",
            llm_url="http://127.0.0.1:11434",
            llm_model="qwen2.5:7b",
            image_steps="invalid_steps",
            image_cfg="invalid_cfg",
            seed="123",
            control_after_generate="fixed",
        )
        self.assertIsInstance(sb, dict)
        self.assertEqual(imgs.shape[0], 2)

    def test_dynamic_shot_durations_and_embellishment(self):
        from director.storyboard import calculate_cinematic_shot_durations, generate_local_storyboard

        # 1. Test calculation of durations with max shot ceiling
        durations = calculate_cinematic_shot_durations(total_duration=15.0, max_shot_duration=15.0, num_beats=4)
        self.assertGreater(len(durations), 1)
        self.assertAlmostEqual(sum(durations), 15.0, places=1)
        for d in durations:
            self.assertLessEqual(d, 15.0)
            self.assertGreaterEqual(d, 2.0)

        # 2. Test generation embellishment with user story
        user_story = "Un chevalier cyberpunk explore les ruelles d'une ville néon sous la pluie battante. Il rencontre un renard holographique lumineux."
        sb_data = generate_local_storyboard(
            story=user_story,
            total_duration=15.0,
            shot_duration=8.0,
            style_prompt="8k, cinematic",
        )
        self.assertGreaterEqual(len(sb_data.scenes), 2)
        # Check that scenes are enriched with atmospheric and spatial details
        first_scene = sb_data.scenes[0]
        self.assertIn("<Subject 2>", first_scene.visual_prompt)
        self.assertIn("<Picture 2>", first_scene.visual_prompt)
        self.assertIn("<Subject 1>", first_scene.visual_prompt)
        # Check durations sum
        scene_dur_sum = sum(s.duration for s in sb_data.scenes)
        self.assertAlmostEqual(scene_dur_sum, 15.0, places=1)
        for s in sb_data.scenes:
            self.assertLessEqual(s.duration, 8.0)
            self.assertGreaterEqual(s.duration, 2.0)


class TestStoryboardHttpRoute(unittest.IsolatedAsyncioTestCase):
    async def test_minimax_storyboard_generate_route(self):
        from director.http_routes import minimax_storyboard_generate

        class MockRequest:
            async def json(self):
                return {
                    "story": "Un pilote dans son mécha sous la pluie nocturne.",
                    "total_duration_sec": 12.0,
                    "shot_duration_sec": 4.0,
                    "aspect_ratio": "16:9 (864x480)",
                    "director_mode": "mixed",
                    "style_prompt": "cinematic anime, 8k",
                    "negative_prompt": "blurry",
                    "ai_story_engine": "Built-in AI Story Engine (Local / No Server needed)",
                    "image_steps": 4,
                    "image_cfg": 2.0,
                    "seed": 42,
                    "node_id": "test_node_1",
                }

        resp = await minimax_storyboard_generate(MockRequest())
        self.assertEqual(resp.status, 200)
        data = json.loads(resp.text)
        self.assertTrue(data.get("ok"))
        self.assertIn("timeline_data", data)
        self.assertIn("prompt", data)
        self.assertGreaterEqual(len(data.get("scenes", [])), 2)
        self.assertGreaterEqual(len(data.get("subjects", [])), 2)
        self.assertEqual(len(data.get("reference_images", [])), len(data.get("subjects", [])))



if __name__ == "__main__":
    unittest.main()
