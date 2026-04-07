import re
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
)
from sglang.srt.models.eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import get_device
from sglang.srt.utils.common import load_video
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger


class Eagle2_5_VLProcessor(BaseMultimodalProcessor):
    models = [Eagle2_5_VLForConditionalGeneration]

    IMG_START = "<img>"
    IMG_END = "</img>"
    IMG_CONTEXT = "<IMG_CONTEXT>"
    IMAGE_PLACEHOLDER = "<|image|>"

    # HF chat_template placeholders
    HF_IMAGE_PLACEHOLDER_RE = re.compile(r"<image-\d+>")
    HF_VIDEO_PLACEHOLDER_RE = re.compile(r"<video-\d+>")

    # vision-id tags: "<image 1>", "<video 2>"
    HF_IMAGE_VISION_ID_RE = re.compile(r"<image\s+\d+>")
    HF_VIDEO_VISION_ID_RE = re.compile(r"<video\s+\d+>")

    # legacy SGLang placeholder
    LEGACY_IMG_PLACEHOLDER_RE = re.compile(r"<img>\s*image\s*</img>")

    VIDEO_PLACEHOLDER = "<|video|>"

    IMAGE_MAX_NUM = 12
    VIDEO_MAX_NUM = 1

    VIDEO_ORDINAL_MAP = {
        0: "first",
        1: "second",
        2: "third",
        3: "fourth",
        4: "fifth",
        5: "sixth",
        6: "seventh",
        7: "eighth",
        8: "ninth",
        9: "tenth",
    }

    # SigLIP2 normalization (not ImageNet)
    SIGLIP2_MEAN = [0.5, 0.5, 0.5]
    SIGLIP2_STD = [0.5, 0.5, 0.5]

    CONTEXT_RESERVED = 256
    CONTEXT_FALLBACK = 40960

    @staticmethod
    @lru_cache(maxsize=32)
    def _get_target_ratios(min_num: int, max_num: int):
        return tuple(
            sorted(
                {
                    (i, j)
                    for n in range(min_num, max_num + 1)
                    for i in range(1, n + 1)
                    for j in range(1, n + 1)
                    if min_num <= i * j <= max_num
                },
                key=lambda x: x[0] * x[1],
            )
        )

    @staticmethod
    @lru_cache(maxsize=4)
    def _build_tile_transform():
        import torchvision.transforms as T

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.ToTensor(),
                T.Normalize(
                    mean=Eagle2_5_VLProcessor.SIGLIP2_MEAN,
                    std=Eagle2_5_VLProcessor.SIGLIP2_STD,
                ),
            ]
        )

    @staticmethod
    def dynamic_preprocess(
        image: Image.Image,
        image_size: int = 448,
        min_num: int = 1,
        max_num: int = IMAGE_MAX_NUM,
        use_thumbnail: bool = False,
    ) -> list:
        """Tile a PIL image into patches matching the best aspect ratio.
        Performs resize and crop in PIL space to match HF training pipeline.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = Eagle2_5_VLProcessor._get_target_ratios(min_num, max_num)

        best_factor = float("-inf")
        best_ratio = (1, 1)
        area = orig_width * orig_height
        for x, y in target_ratios:
            target_ar = x / y
            factor = min((x * y * image_size * image_size) / area, 0.6) * min(
                target_ar / aspect_ratio, aspect_ratio / target_ar
            )
            if factor > best_factor:
                best_factor = factor
                best_ratio = (x, y)

        target_w = image_size * best_ratio[0]
        target_h = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tiles = []
        for i in range(blocks):
            x0 = (i % best_ratio[0]) * image_size
            y0 = (i // best_ratio[0]) * image_size
            tile = resized.crop((x0, y0, x0 + image_size, y0 + image_size))
            tiles.append(tile)

        if use_thumbnail and len(tiles) > 1:
            tiles.append(
                image.resize((image_size, image_size), Image.Resampling.BICUBIC)
            )

        return tiles

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.hf_config = hf_config

        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        self.tokenizer = tokenizer
        image_processor = getattr(self._processor, "image_processor", None)

        image_size = (
            getattr(hf_config, "force_image_size", None)
            or hf_config.vision_config.image_size
        )
        patch_size = hf_config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        self.image_size = image_size
        downsample_ratio = getattr(hf_config, "downsample_ratio", 0.5)
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (downsample_ratio**2)
        )

        # Video reuses <IMG_CONTEXT>
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT)
        if self.img_context_token_id is None or self.img_context_token_id < 0:
            raise RuntimeError("Token <IMG_CONTEXT> not found in tokenizer vocab.")
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END)
        self.video_token_id = self.img_context_token_id
        self.default_min_dynamic_patch = getattr(
            hf_config,
            "min_dynamic_tiles",
            getattr(image_processor, "min_dynamic_tiles", 1),
        )
        self.default_max_dynamic_patch = getattr(
            hf_config,
            "max_dynamic_tiles",
            getattr(image_processor, "max_dynamic_tiles", self.IMAGE_MAX_NUM),
        )
        self.use_thumbnail = bool(
            getattr(
                hf_config,
                "use_thumbnail",
                getattr(image_processor, "use_thumbnail", False),
            )
        )
        self.max_context_len = (
            getattr(server_args, "context_length", None)
            or getattr(server_args, "max_context_len", None)
            or getattr(hf_config, "max_position_embeddings", None)
            or getattr(
                getattr(hf_config, "text_config", None), "max_position_embeddings", None
            )
            or self.CONTEXT_FALLBACK
        )

        # image_token differs from image_token_id: split token vs offset token
        # NOTE: video_token intentionally omitted — image and video share
        # img_context_token_id, so registering video_token would cause
        # load_mm_data to match <|video|> placeholders without a data iterator
        # (the normal path handles video manually after load_mm_data).
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_PLACEHOLDER,
            image_token_id=self.img_context_token_id,
        ).build(_processor)

        logger.debug(
            "[Eagle2.5-VL] processor init: num_image_token=%d, img_context_token_id=%d",
            self.num_image_token,
            self.img_context_token_id,
        )

    # Placeholder normalization
    def _rewrite_mm_placeholders(self, input_text: str) -> str:
        if not isinstance(input_text, str):
            return input_text

        # Remove vision-id tags
        input_text = self.HF_IMAGE_VISION_ID_RE.sub("", input_text)
        input_text = self.HF_VIDEO_VISION_ID_RE.sub("", input_text)

        # Normalize all image placeholder formats → <|image|>
        input_text = self.HF_IMAGE_PLACEHOLDER_RE.sub(
            self.IMAGE_PLACEHOLDER, input_text
        )
        input_text = self.LEGACY_IMG_PLACEHOLDER_RE.sub(
            self.IMAGE_PLACEHOLDER, input_text
        )

        # Normalize video placeholders → <|video|>
        input_text = self.HF_VIDEO_PLACEHOLDER_RE.sub(
            self.VIDEO_PLACEHOLDER, input_text
        )

        return input_text

    # Placeholder balancing
    @staticmethod
    def _ensure_placeholders_before_assistant(
        prompt: str, placeholder: str, want: int
    ) -> str:
        if want <= 0:
            return prompt
        have = (prompt or "").count(placeholder)
        missing = want - have
        if missing <= 0:
            return prompt

        insert = "\n" + "\n".join([placeholder] * missing) + "\n"

        # Try both assistant markers: Eagle uses <|assistant|>, InternVL uses <|im_start|>assistant
        for marker in ("<|assistant|>", "<|im_start|>assistant"):
            idx = (prompt or "").rfind(marker)
            if idx != -1:
                return (prompt or "")[:idx] + insert + (prompt or "")[idx:]
        return (prompt or "") + insert

    # Special format delegation (processor_output / precomputed_embedding)
    @staticmethod
    def _has_special_format(image_data, video_data=None):
        for data in list(image_data or []) + list(video_data or []):
            if isinstance(data, dict) and data.get("format") in (
                "processor_output",
                "precomputed_embedding",
            ):
                return True
        return False

    async def _process_special_format(self, image_data, video_data, input_text):
        if video_data:
            raise NotImplementedError(
                "[Eagle2.5-VL] Video is not supported in the special-format path "
                "(processor_output / precomputed_embedding / pre-tokenized input_ids). "
                "Image and video share the same context token, so offset "
                "disambiguation requires dedicated logic. "
                "Use raw video inputs with a text prompt instead."
            )

        if isinstance(input_text, list):
            user_input_ids = input_text
            prompt = ""
        else:
            user_input_ids = None
            prompt = input_text or ""

        if not prompt and image_data:
            images = [d for d in image_data if isinstance(d, dict)]
            raw_dropped = len(image_data) - len(images)
            if raw_dropped > 0:
                raise ValueError(
                    f"[Eagle2.5-VL] Cannot process raw images with pre-tokenized "
                    f"input_ids. Use 'processor_output' or 'precomputed_embedding' "
                    f"format, or provide a text prompt. "
                    f"(raw images dropped: {raw_dropped})"
                )
            base_output = BaseMultiModalProcessorOutput(
                input_text=prompt,
                images=images,
            )
        else:
            prompt = self._rewrite_mm_placeholders(prompt)
            base_output = self.load_mm_data(
                prompt=prompt,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=True,
            )

        mm_items, input_ids_tensor, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # Override with user-provided input_ids and recompute offsets
        if user_input_ids is not None:
            input_ids_tensor = torch.tensor(user_input_ids, dtype=torch.long)
            all_offsets = self.get_mm_items_offset(
                input_ids=input_ids_tensor,
                mm_token_id=self.img_context_token_id,
            )
            if len(mm_items) == 1:
                # Single item holds all images — assign all spans
                mm_items[0].offsets = all_offsets
            else:
                if len(all_offsets) != len(mm_items):
                    raise ValueError(
                        f"[Eagle2.5-VL] Offset span count ({len(all_offsets)}) != "
                        f"mm_item count ({len(mm_items)}) for pre-tokenized input_ids."
                    )
                for i, mm_item in enumerate(mm_items):
                    mm_item.offsets = [all_offsets[i]]

        return {
            "input_ids": input_ids_tensor.flatten().tolist(),
            "mm_items": mm_items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
            "video_token_id": self.video_token_id,
        }

    def _tiles_to_tensor(self, pil_tiles: list) -> torch.Tensor:
        transform = self._build_tile_transform()
        return torch.stack([transform(t) for t in pil_tiles]).to(
            dtype=torch.bfloat16, device=get_device()
        )

    def _preprocess_images(self, images, min_tiles=None, max_tiles=None):
        if min_tiles is None:
            min_tiles = self.default_min_dynamic_patch
        if max_tiles is None:
            max_tiles = self.default_max_dynamic_patch

        num_patches_list = []
        pixel_values_list = []

        for image in images:
            pil_tiles = self.dynamic_preprocess(
                image,
                image_size=self.image_size,
                min_num=min_tiles,
                max_num=max_tiles,
                use_thumbnail=self.use_thumbnail,
            )

            tensor_tiles = self._tiles_to_tensor(pil_tiles)
            if not self.server_args.keep_mm_feature_on_device:
                tensor_tiles = tensor_tiles.cpu()

            num_patches_list.append(len(pil_tiles))
            pixel_values_list.append(tensor_tiles)

        return pixel_values_list, num_patches_list

    def _token_len(self, text: str) -> int:
        ids = self.tokenizer(text, return_tensors="pt").input_ids.flatten()
        return int(ids.numel())

    def _resolve_video_num_frames(
        self,
        *,
        requested: int,
        num_videos: int,
        text_len: int,
        image_tile_cnt: int,
        vid_max_tiles: int = 1,
    ) -> int:
        """Cap video frame count to fit within context budget."""
        if num_videos <= 0:
            return 0
        image_tokens = image_tile_cnt * self.num_image_token
        budget = (
            int(self.max_context_len)
            - int(text_len)
            - int(image_tokens)
            - int(self.CONTEXT_RESERVED)
        )
        if budget <= 0:
            return 1
        tokens_per_frame = self.num_image_token * max(1, int(vid_max_tiles))
        max_total_frames = max(1, budget // tokens_per_frame)
        frames_per_video = max(1, max_total_frames // max(num_videos, 1))
        return max(1, min(int(requested), int(frames_per_video)))

    DEFAULT_VIDEO_NUM_FRAMES = 8

    @staticmethod
    def _load_video_input(video_src):
        return load_video(video_src)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        video_data = getattr(request_obj, "video_data", None) or []

        if audio_data:
            raise NotImplementedError("Eagle2_5_VL does not support audio.")

        if isinstance(input_text, list) or self._has_special_format(
            image_data, video_data
        ):
            return await self._process_special_format(
                image_data=image_data,
                video_data=video_data,
                input_text=input_text,
            )

        prompt = input_text or ""
        if not image_data and not video_data:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            ).input_ids.flatten()
            return {
                "input_ids": input_ids.tolist(),
                "mm_items": [],
                "im_start_id": self.img_start_token_id,
                "im_end_id": self.img_end_token_id,
                "im_token_id": self.img_context_token_id,
                "video_token_id": self.video_token_id,
            }

        prompt = self._rewrite_mm_placeholders(prompt)

        img_max_num = (
            getattr(request_obj, "image_max_dynamic_patch", None)
            or getattr(request_obj, "max_dynamic_patch", None)
            or kwargs.get("image_max_dynamic_patch")
            or kwargs.get("max_dynamic_patch")
            or self.default_max_dynamic_patch
        )
        img_max_num = max(1, int(img_max_num))
        img_min_num = max(1, int(self.default_min_dynamic_patch))

        vid_max_num = (
            getattr(request_obj, "video_max_dynamic_patch", None)
            or getattr(request_obj, "max_dynamic_patch", None)
            or kwargs.get("video_max_dynamic_patch")
            or kwargs.get("max_dynamic_patch")
            or self.VIDEO_MAX_NUM
        )
        vid_max_num = max(1, int(vid_max_num))

        if image_data:
            prompt = self._ensure_placeholders_before_assistant(
                prompt, self.IMAGE_PLACEHOLDER, len(image_data)
            )
        if video_data:
            prompt = self._ensure_placeholders_before_assistant(
                prompt, self.VIDEO_PLACEHOLDER, len(video_data)
            )

        img_placeholder_count = prompt.count(self.IMAGE_PLACEHOLDER)
        if image_data and img_placeholder_count != len(image_data):
            raise ValueError(
                f"[Eagle2.5-VL] Image placeholder count ({img_placeholder_count}) != "
                f"image count ({len(image_data)}) after balancing."
            )

        vid_placeholder_count = prompt.count(self.VIDEO_PLACEHOLDER)
        if video_data and vid_placeholder_count != len(video_data):
            raise ValueError(
                f"[Eagle2.5-VL] Video placeholder count ({vid_placeholder_count}) != "
                f"video count ({len(video_data)}) after balancing."
            )

        base_output = self.load_mm_data(
            prompt=prompt,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        pixel_values_list = []
        num_patches_list = []
        if base_output.images:
            pixel_values_list, num_patches_list = self._preprocess_images(
                base_output.images,
                min_tiles=img_min_num,
                max_tiles=img_max_num,
            )

        video_pixel_values_list = []
        video_frame_patch_lists = []
        if video_data:
            requested_frames = int(
                kwargs.get("video_num_frames", self.DEFAULT_VIDEO_NUM_FRAMES)
            )
            num_frames = self._resolve_video_num_frames(
                requested=requested_frames,
                num_videos=len(video_data),
                text_len=self._token_len(prompt),
                image_tile_cnt=sum(num_patches_list) if num_patches_list else 0,
                vid_max_tiles=vid_max_num,
            )
            for video_src in video_data:
                is_video_obj = isinstance(video_src, VideoDecoderWrapper)
                vr = video_src if is_video_obj else self._load_video_input(video_src)
                max_frame = len(vr) - 1
                frame_indices = (
                    [0]
                    if num_frames == 1
                    else np.linspace(0, max_frame, num=num_frames, dtype=int).tolist()
                )

                per_video_tiles = []
                per_frame_patch_cnt = []
                for fi in frame_indices:
                    frame = vr[int(fi)]
                    frame_img = Image.fromarray(frame).convert("RGB")
                    pil_tiles = self.dynamic_preprocess(
                        frame_img,
                        image_size=self.image_size,
                        min_num=1,
                        max_num=vid_max_num,
                        use_thumbnail=self.use_thumbnail,
                    )
                    per_video_tiles.append(self._tiles_to_tensor(pil_tiles))
                    per_frame_patch_cnt.append(len(pil_tiles))

                video_pv = torch.cat(per_video_tiles, dim=0)
                if not self.server_args.keep_mm_feature_on_device:
                    video_pv = video_pv.cpu()
                video_pixel_values_list.append(video_pv)
                video_frame_patch_lists.append(per_frame_patch_cnt)

        img_ph = "<<<__EAGLE_IMG_PH__>>>"
        vid_ph = "<<<__EAGLE_VID_PH__>>>"
        text = base_output.input_text
        text = text.replace(self.IMAGE_PLACEHOLDER, img_ph)
        text = text.replace(self.VIDEO_PLACEHOLDER, vid_ph)

        # Image and video both expand to <IMG_CONTEXT> tokens (model trained this way reference: https://huggingface.co/nvidia/Eagle2.5-8B/blob/main/processing_eagle2_5_vl.py)
        # so after tokenization we can't distinguish them by token_id alone.
        # Record the prompt-order of placeholders BEFORE expansion so we can
        # split the flat offset list into image vs video spans later
        modality_order = []  # list of ("image", idx) or ("video", idx)
        scan = text
        img_idx = 0
        vid_idx = 0
        while True:
            img_pos = scan.find(img_ph)
            vid_pos = scan.find(vid_ph)
            if img_pos < 0 and vid_pos < 0:
                break
            if vid_pos < 0 or (img_pos >= 0 and img_pos < vid_pos):
                modality_order.append(("image", img_idx))
                img_idx += 1
                scan = scan.replace(img_ph, "", 1)
            else:
                modality_order.append(("video", vid_idx))
                vid_idx += 1
                scan = scan.replace(vid_ph, "", 1)

        # Expand image placeholders
        for expand_idx, num_patches in enumerate(num_patches_list):
            expanded = (
                f"<image {expand_idx + 1}>"
                + self.IMG_START
                + self.IMG_CONTEXT * (num_patches * self.num_image_token)
                + self.IMG_END
            )
            text = text.replace(img_ph, expanded, 1)

        # Expand video placeholders (each frame is a <img>...</img> block)
        for video_idx, frame_patch_list in enumerate(video_frame_patch_lists):
            frame_lines = []
            for i, patch_cnt in enumerate(frame_patch_list):
                ctx_cnt = patch_cnt * self.num_image_token
                frame_tokens = (
                    self.IMG_START + self.IMG_CONTEXT * ctx_cnt + self.IMG_END
                )
                frame_lines.append(f"Frame {i+1}: {frame_tokens}")
            ordinal = self.VIDEO_ORDINAL_MAP.get(video_idx, f"#{video_idx+1}")
            video_tokens = f"The {ordinal} video: " + "".join(frame_lines)
            text = text.replace(vid_ph, video_tokens, 1)

        if img_ph in text or vid_ph in text:
            raise ValueError(
                "[Eagle2.5-VL] Unexpanded placeholders remain after expansion."
            )

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.flatten()

        all_offsets = self.get_mm_items_offset(
            input_ids=input_ids, mm_token_id=self.img_context_token_id
        )

        num_video_spans = sum(len(fl) for fl in video_frame_patch_lists)
        expected_total = len(num_patches_list) + num_video_spans
        if len(all_offsets) != expected_total:
            raise ValueError(
                f"[Eagle2.5-VL] Total offset spans ({len(all_offsets)}) != "
                f"expected ({expected_total}: {len(num_patches_list)} images + "
                f"{num_video_spans} video frames)."
            )

        image_offsets = [None] * len(num_patches_list)
        video_offsets_flat = []
        offset_idx = 0
        for modality, idx in modality_order:
            if modality == "image":
                image_offsets[idx] = all_offsets[offset_idx]
                offset_idx += 1
            else:
                num_frames = len(video_frame_patch_lists[idx])
                for _ in range(num_frames):
                    video_offsets_flat.append(all_offsets[offset_idx])
                    offset_idx += 1

        image_offsets = [o for o in image_offsets if o is not None]
        video_offsets = video_offsets_flat

        # Per-image span assertion
        for i, (start, end) in enumerate(image_offsets):
            expected_len = num_patches_list[i] * self.num_image_token
            actual_len = end - start + 1
            if actual_len != expected_len:
                raise ValueError(
                    f"[Eagle2.5-VL] Image {i}: offset span {actual_len} != "
                    f"expected {expected_len} "
                    f"({num_patches_list[i]} tiles * {self.num_image_token} tokens)."
                )

        items = []
        if pixel_values_list:
            items.append(
                MultimodalDataItem(
                    feature=torch.cat(pixel_values_list, dim=0),
                    modality=Modality.IMAGE,
                    offsets=image_offsets,
                )
            )
        if video_pixel_values_list:
            items.append(
                MultimodalDataItem(
                    feature=torch.cat(video_pixel_values_list, dim=0),
                    modality=Modality.VIDEO,
                    offsets=video_offsets,
                )
            )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
            "video_token_id": self.video_token_id,
        }
