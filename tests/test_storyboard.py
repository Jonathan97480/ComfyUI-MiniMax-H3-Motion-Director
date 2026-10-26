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
    StoryboardScene,
    build_director_timeline,
    format_script_summary,
    generate_local_scenes,
    generate_scene_reference_images,
    parse_aspect_ratio,
    render_storyboard_visual_concept,
)
from nodes.storyboard_node import MiniMaxH3StoryDirector


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

    def test_generate_local_scenes(self):
        story = (
            "Un ninja cybernétique s'infiltre dans une forteresse volante au-dessus des nuages. "
            "Il neutralise un drone de surveillance avec précision. "
            "Il accède au terminal central et télécharge les données secrètes."
        )
        scenes = generate_local_scenes(
            story=story,
            num_scenes=3,
            total_duration=15.0,
            style_prompt="cinematic, 8k",
        )
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0].duration, 5.0)
        self.assertFalse(scenes[0].continuity_visual)
        self.assertTrue(scenes[1].continuity_visual)
        self.assertTrue(scenes[2].continuity_visual)
        self.assertTrue(len(scenes[0].full_prompt) > 20)
        self.assertTrue(len(scenes[0].image_prompt) > 20)

    def test_render_storyboard_visual_concept(self):
        tensor = render_storyboard_visual_concept(
            scene_num=1,
            total_scenes=3,
            title="Plan 1 : Exposition",
            story_snippet="A cybernetic knight explores rain-slicked neon streets",
            lighting="Volumetric cyan and magenta neon",
            width=864,
            height=480,
            seed=42,
        )
        self.assertEqual(tensor.shape, (1, 480, 864, 3))
        self.assertTrue(torch.is_floating_point(tensor))
        self.assertTrue(tensor.min() >= 0.0)
        self.assertTrue(tensor.max() <= 1.0)

    def test_build_director_timeline(self):
        scenes = generate_local_scenes(
            story="Test story",
            num_scenes=2,
            total_duration=10.0,
            style_prompt="photorealistic",
        )
        scenes[0].image_b64 = "data:image/jpeg;base64,abc"
        scenes[1].image_b64 = "data:image/jpeg;base64,def"

        timeline = build_director_timeline(
            scenes,
            director_mode="mixed",
            width=864,
            height=480,
            fps=24.0,
        )
        self.assertEqual(timeline["timelineMode"], "mixed")
        self.assertEqual(len(timeline["segments"]), 2)
        self.assertEqual(timeline["segments"][0]["duration"], 5.0)
        self.assertEqual(timeline["segments"][0]["inputs"]["startFrame"]["imageB64"], "data:image/jpeg;base64,abc")

    def test_node_execution(self):
        node = MiniMaxH3StoryDirector()
        sb, t_json, imgs, summary = node.execute(
            story="Le samouraï dégaine son katana énergétique. La pluie s'arrête net.",
            total_duration_sec=10.0,
            shot_duration_sec=5.0,
            aspect_ratio="16:9 (864x480)",
            director_mode="mixed",
            style_prompt="8k, masterpiece",
            negative_prompt="blurry, text",
            ai_story_engine="Built-in AI Story Engine (Local / No Server needed)",
            llm_url="http://127.0.0.1:11434",
            llm_model="qwen2.5:7b",
            image_steps=20,
            image_cfg=6.5,
            seed=123,
            control_after_generate="fixed",
        )
        self.assertIsInstance(sb, dict)
        self.assertEqual(sb["version"], 1)
        self.assertEqual(len(sb["scenes"]), 2)
        self.assertEqual(imgs.shape, (2, 480, 864, 3))
        self.assertIn("timelineMode", t_json)
        self.assertIn("SCÉNARIO", summary)

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
        self.assertEqual(len(sb["scenes"]), 2)


if __name__ == "__main__":
    unittest.main()
