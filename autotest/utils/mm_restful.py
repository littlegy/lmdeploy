"""Lightweight multimodal helpers for REST protocol suites (no lmdeploy
import)."""

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
    """Protocol-only multimodal cases (no tools): native VL model_case
    names."""
    if not is_mm_tool_call_capable(model_case):
        pytest.skip(
            f'model {model_case!r} is not treated as vision-capable for multimodal protocol tests')


def is_mm_video_unsupported(model_case: str) -> bool:
    """Intern-S1 family rejects type=video; image MM still runs."""
    return 'intern-s1' in model_case.lower()


def resolve_mm_resource_path(config, filename: str) -> str | None:
    """Return a local filesystem path for a test media file, or None if
    missing."""
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


def chat_content_parts_to_responses_parts(parts: list[dict]) -> list[dict]:
    """Chat ``messages[*].content`` parts → Responses ``input`` content
    parts."""
    out: list[dict] = []
    for part in parts:
        pt = part.get('type')
        if pt == 'text':
            out.append({'type': 'input_text', 'text': part['text']})
        elif pt in ('image_url', 'image'):
            raw = part.get('image_url')
            if isinstance(raw, dict):
                url = raw.get('url')
            elif isinstance(raw, str):
                url = raw
            else:
                url = None
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f'missing image url in chat content part {part!r}')
            out.append({'type': 'input_image', 'image_url': url})
        else:
            raise ValueError(f'unsupported chat content part for Responses: {pt!r}')
    return out


def chat_messages_to_responses_input(messages: list[dict]) -> list[dict]:
    """Chat-style ``messages`` → Responses API ``input`` (message items)."""
    items: list[dict] = []
    for msg in messages:
        if msg.get('type') in ('function_call', 'function_call_output'):
            items.append(msg)
            continue
        role = msg.get('role')
        if role is None:
            raise ValueError(f'message missing role: {msg!r}')
        content = msg.get('content', '')
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get('type') in ('image_url', 'image')
            for p in content
        ):
            content = chat_content_parts_to_responses_parts(content)
        item = dict(msg)
        item['content'] = content
        item.setdefault('type', 'message')
        items.append(item)
    return items


def responses_user_image_input(*, image_url: str, text: str) -> list[dict]:
    """Single user turn with text + image (Responses ``input``)."""
    return chat_messages_to_responses_input([{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {'url': image_url}},
        ],
    }])


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
