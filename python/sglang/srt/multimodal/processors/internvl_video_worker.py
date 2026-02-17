"""InternVL video frame extraction worker.

Runs in ProcessPoolExecutor (spawn context) — must be importable with
minimal dependencies and contain no CUDA state at module level.
"""

import numpy as np

from sglang.srt.multimodal.processors.video_utils import VideoInput


def _get_decord_context():
    """Probe GPU availability for decord, fall back to CPU."""
    from decord import cpu, gpu

    try:
        from decord.bridge import decord_bridge

        ctx = gpu(0)
        _ = decord_bridge.get_ctx_device(ctx)
        return ctx
    except Exception:
        return cpu(0)


def extract_video_frames(
    video_input,
    num_frames: int,
) -> list:
    """Extract *num_frames* frames from a video as numpy arrays.

    Designed to run in a subprocess via ``ProcessPoolExecutor(spawn)``.
    Opens the video, samples frames uniformly, and returns a list of
    ``np.ndarray`` in (H, W, C) uint8 format.

    GPU normalisation and tiling are left to the caller so this worker
    stays CUDA-free.
    """
    from decord import VideoReader

    if isinstance(video_input, VideoInput):
        path = video_input.path
    elif isinstance(video_input, str):
        path = video_input
    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")

    ctx = _get_decord_context()
    vr = VideoReader(path, ctx=ctx, num_threads=1)

    max_frame = len(vr) - 1
    if num_frames <= 1:
        frame_indices = np.array([0], dtype=np.int64)
    else:
        frame_indices = np.linspace(0, max_frame, num=num_frames, dtype=np.int64)
        frame_indices = np.unique(frame_indices)

    # Batch read is significantly faster than per-frame indexing
    batch = vr.get_batch(frame_indices)
    frames_np = batch.asnumpy()  # (T, H, W, C)
    frames = [frames_np[i] for i in range(frames_np.shape[0])]

    # TODO: clean up temp file if video_input.cleanup is True
    return frames
