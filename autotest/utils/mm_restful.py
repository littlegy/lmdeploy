"""Lightweight multimodal helpers for REST protocol suites (no lmdeploy import)."""

from __future__ import annotations

import base64
import mimetypes
import os

import pytest

# Test media filenames under config['resource_path'].
MM_TEST_IMAGE_TIGER = 'tiger.jpeg'
MM_TEST_IMAGE_BEIJING = 'Beijing_Small.jpeg'
MM_TEST_IMAGE_POSE = 'human-pose.jpg'
MM_TEST_VIDEO = 'red-panda.mp4'

_MM_TOOL_CALL_MODEL_MARKERS = (
    'Qwen3.5',
    'Intern-S1',
    'Intern-S2',
    'Qwen3-VL',
)


def is_mm_tool_call_capable(model_case: str) -> bool:
    """True for model families that support image input and tool calling."""
    name = model_case.lower()
    return any(marker.lower() in name for marker in _MM_TOOL_CALL_MODEL_MARKERS)


def skip_unless_vlm_model(model_case: str) -> None:
    """Protocol-only multimodal cases (no tools): native VL model_case names."""
    if not is_mm_tool_call_capable(model_case):
        pytest.skip(
            f'model {model_case!r} is not treated as vision-capable for multimodal protocol tests')


def is_mm_video_unsupported(model_case: str) -> bool:
    """Intern-S1 family rejects type=video; image MM still runs."""
    return 'intern-s1' in model_case.lower()


def resolve_mm_resource_path(config, filename: str) -> str | None:
    """Return a local filesystem path for a test media file, or None if missing."""
    path = os.path.join(config['resource_path'], filename)
    if os.path.isfile(path):
        return path
    return None


def resolve_mm_image_url(config, filename: str) -> str | None:
    """Return a local filesystem path for a test image, or None if missing."""
    return resolve_mm_resource_path(config, filename)


def require_mm_image_url(config, filename: str = MM_TEST_IMAGE_TIGER) -> str:
    """Local image under ``config['resource_path']`` or pytest.skip."""
    image_url = resolve_mm_image_url(config, filename)
    if image_url is None:
        pytest.skip(f'Missing multimodal image {filename!r} under resource_path')
    return image_url


def mm_file_to_data_url(path: str, *, mime: str | None = None) -> str:
    """Encode a local media file as a ``data:<mime>;base64,...`` URL."""
    if mime is None:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or 'application/octet-stream'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def build_multimodal_media_part(
        media_type: str,
        source,
        **fields) -> dict:
    """Build one OpenAI-style multimodal content part for *media_type*."""
    if media_type in ('image_url', 'image', 'video_url', 'video'):
        return {media_type: {'url': source, **fields}, 'type': media_type}
    raise ValueError(f'Unsupported multimodal media type: {media_type!r}')


def build_multimodal_user_message(
        text: str,
        media: str | list[str] | list[tuple[str, object]],
        *,
        media_type: str = 'image_url',
        media_first: bool = False,
        **media_fields) -> dict:
    """OpenAI-compatible user message with text + one or more media parts."""
    if isinstance(media, str):
        pairs: list[tuple[str, object]] = [(media_type, media)]
    elif media and isinstance(media[0], tuple):
        pairs = list(media)  # type: ignore[arg-type]
    else:
        pairs = [(media_type, src) for src in media]  # type: ignore[union-attr]

    media_parts = [
        build_multimodal_media_part(mt, src, **media_fields)
        for mt, src in pairs
    ]
    text_part = {'type': 'text', 'text': text}
    parts = media_parts + [text_part] if media_first else [text_part] + media_parts
    return {'role': 'user', 'content': parts}


def mm_media_fixture_filename(media_type: str) -> str:
    """Map a media type to a filename under resource_path."""
    if media_type in ('image_url', 'image'):
        return MM_TEST_IMAGE_TIGER
    if media_type in ('video_url', 'video'):
        return MM_TEST_VIDEO
    raise ValueError(f'No local fixture for media type {media_type!r}')