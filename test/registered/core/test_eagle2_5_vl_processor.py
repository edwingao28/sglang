"""Unit tests for Eagle2_5_VLProcessor token expansion and offset extraction.

These tests validate the preprocessing pipeline without loading the full model.
They run on CPU and require only the tokenizer (downloaded on first run).
"""

import unittest


from transformers import AutoConfig

from sglang.test.ci.ci_register import register_cuda_ci

MODEL_PATH = "nvidia/Eagle2.5-8B"

register_cuda_ci(est_time=60, suite="stage-b-test-large-1-gpu")


class TestEagle2_5_VLProcessorTokenExpansion(unittest.TestCase):
    """Validate that image placeholders are expanded correctly."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        cls.config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )

    def test_special_tokens_in_vocab(self):
        """Critical tokens must exist in the tokenizer vocabulary."""
        vocab = self.tokenizer.get_vocab()
        for token in ["<IMG_CONTEXT>", "<img>", "</img>", "<|im_start|>", "<|im_end|>"]:
            self.assertIn(token, vocab, f"Token {token!r} missing from vocab")

    def test_img_context_token_id_matches_config(self):
        """<IMG_CONTEXT> token ID must match config.image_token_index."""
        vocab = self.tokenizer.get_vocab()
        actual_id = vocab["<IMG_CONTEXT>"]
        expected_id = self.config.image_token_index
        self.assertEqual(actual_id, expected_id)

    def test_num_image_token_formula(self):
        """Verify the token-count formula: (image_size/patch_size)^2 * downsample_ratio^2."""
        image_size = self.config.vision_config.image_size  # 448
        patch_size = self.config.vision_config.patch_size  # 14
        downsample_ratio = self.config.downsample_ratio  # 0.5
        expected = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
        self.assertEqual(expected, 256)

    def test_chat_template_no_unk_tokens(self):
        """Applying the chat template must not produce UNK tokens."""
        prompt = (
            "<|im_start|>user\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.flatten()
        unk_id = self.tokenizer.unk_token_id
        if unk_id is not None:
            unk_count = (input_ids == unk_id).sum().item()
            self.assertEqual(unk_count, 0, f"Found {unk_count} UNK tokens in: {prompt}")


class TestEagle2_5_VLTemplateRouting(unittest.TestCase):
    """Eagle 2.5 must NOT route to InternVL conversation template."""

    def test_model_type_not_in_internvl_mapping(self):
        """eagle_2_5_vl must not appear in MODEL_TYPE_TO_TEMPLATE."""
        from sglang.srt.parser.conversation import MODEL_TYPE_TO_TEMPLATE

        self.assertNotIn(
            "eagle_2_5_vl",
            MODEL_TYPE_TO_TEMPLATE,
            "eagle_2_5_vl must not map to internvl-2-5; it has its own template",
        )

    def test_chat_template_match_returns_eagle(self):
        """match_chat_ml must return eagle25-vl for Eagle 2.5 model paths."""
        from sglang.lang.chat_template import match_chat_ml

        for path in ["nvidia/Eagle2.5-8B", "models/eagle2.5-vl-chat", "Eagle2.5"]:
            result = match_chat_ml(path)
            self.assertEqual(result, "eagle25-vl", f"Failed for path: {path}")


class TestEagle2_5_VLVideoBudget(unittest.TestCase):
    """Video budget must account for multi-tile frames."""

    def test_budget_scales_with_tiles_per_frame(self):
        """With max_tiles=4, fewer frames should be allowed than max_tiles=1."""
        from sglang.srt.multimodal.processors.eagle2_5_vl import Eagle2_5_VLProcessor

        # Create a minimal processor mock with required attributes
        proc = Eagle2_5_VLProcessor.__new__(Eagle2_5_VLProcessor)
        proc.num_image_token = 256
        proc.max_context_len = 4096
        proc.CONTEXT_RESERVED = 256

        # With 1 tile/frame: budget = (4096 - 100 - 0 - 256) / (256*1) = 14 frames
        frames_1tile = proc._resolve_video_num_frames(
            requested=32,
            num_videos=1,
            text_len=100,
            image_tile_cnt=0,
            vid_max_tiles=1,
        )

        # With 4 tiles/frame: budget = (4096 - 100 - 0 - 256) / (256*4) = 3 frames
        frames_4tile = proc._resolve_video_num_frames(
            requested=32,
            num_videos=1,
            text_len=100,
            image_tile_cnt=0,
            vid_max_tiles=4,
        )

        self.assertGreater(frames_1tile, frames_4tile)
        # 4 tiles should allow roughly 1/4 the frames
        self.assertLessEqual(frames_4tile, frames_1tile // 3)


class TestEagle2_5_VLPadInputIds(unittest.TestCase):
    """pad_input_ids must assign distinct pad_values per <img>...</img> block."""

    # Token IDs (arbitrary but consistent)
    IMG_START = 100
    IMG_END = 101
    IMG_CONTEXT = 200  # shared by both image and video
    TEXT = 50

    def _build_input_ids(self, blocks):
        """Build input_ids from a list of (num_context_tokens,) per block.

        Each block becomes: [IMG_START, IMG_CONTEXT * n, IMG_END]
        Separated by a TEXT token.
        """
        ids = []
        for i, n in enumerate(blocks):
            if i > 0:
                ids.append(self.TEXT)
            ids.append(self.IMG_START)
            ids.extend([self.IMG_CONTEXT] * n)
            ids.append(self.IMG_END)
        return ids

    def test_mixed_image_video_gets_distinct_pad_values(self):
        """Two blocks sharing IMG_CONTEXT must get different pad_values."""

        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternTokenPairs,
        )
        from sglang.srt.managers.schedule_batch import (
            Modality,
            MultimodalDataItem,
            MultimodalInputs,
        )

        # 1 image block (4 tokens) + 1 video block (3 tokens)
        input_ids = self._build_input_ids([4, 3])
        # input_ids: [100, 200,200,200,200, 101, 50, 100, 200,200,200, 101]

        image_item = MultimodalDataItem(modality=Modality.IMAGE, pad_value=-100)
        video_item = MultimodalDataItem(modality=Modality.VIDEO, pad_value=-200)

        mm_inputs = MultimodalInputs(
            mm_items=[image_item, video_item],
            im_token_id=self.IMG_CONTEXT,
            im_start_id=self.IMG_START,
            im_end_id=self.IMG_END,
            video_token_id=self.IMG_CONTEXT,  # same as im_token_id!
        )

        pattern = MultiModalityDataPaddingPatternTokenPairs(
            [(self.IMG_START, self.IMG_END)]
        )
        padded = pattern.pad_input_tokens(input_ids, mm_inputs)

        self.assertEqual(len(padded), len(input_ids))

        # Block 0 (image): indices 1-4 should be -100
        self.assertEqual(padded[1:5], [-100] * 4)
        # Block 1 (video): indices 8-10 should be -200
        self.assertEqual(padded[8:11], [-200] * 3)
        # Boundary/text tokens unchanged
        self.assertEqual(padded[0], self.IMG_START)
        self.assertEqual(padded[5], self.IMG_END)
        self.assertEqual(padded[6], self.TEXT)


if __name__ == "__main__":
    unittest.main()
