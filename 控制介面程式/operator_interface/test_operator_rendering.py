"""Focused tests for the pure video rendering helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from operator_rendering import draw_video_hud  # noqa: E402


class _RecordingDraw:
    def __init__(self) -> None:
        self.rectangles: list[tuple[tuple[int, ...], dict]] = []
        self.texts: list[tuple[tuple[int, int], str, dict]] = []

    def rectangle(self, box, **kwargs) -> None:
        self.rectangles.append((tuple(box), kwargs))

    def text(self, xy, text, **kwargs) -> None:
        self.texts.append((tuple(xy), text, kwargs))


def test_video_hud_draws_black_text_directly_on_stream_without_bottom_panel() -> None:
    draw = _RecordingDraw()
    state = SimpleNamespace(
        frame_age_ms=12.4,
        stream_fps=30.0,
        gimbal_pitch_deg=-20.0,
        zoom=1.0,
        link_ok=False,
    )

    link_ok = draw_video_hud(
        draw,
        320,
        240,
        state,
        ("diagnostic one", "diagnostic two"),
        60,
        object(),
        object(),
        object(),
        True,
        280.0,
    )

    assert link_ok is False
    assert draw.rectangles == [
        ((18, 18, 302, 222), {"outline": "#363c44", "width": 2}),
    ]
    assert [text for _, text, _ in draw.texts] == [
        "影像 30.0 FPS · 影格新鮮度 12 ms · 鏡頭 -20° / 1.0x",
        "diagnostic one",
        "diagnostic two",
    ]
    assert [xy for xy, _, _ in draw.texts] == [(28, 185), (28, 202), (28, 219)]
    assert all(kwargs["fill"] == "#000000" for _, _, kwargs in draw.texts)
    assert all(kwargs["stroke_width"] == 1 for _, _, kwargs in draw.texts)
    assert all(kwargs["stroke_fill"] == "#f4f4f4" for _, _, kwargs in draw.texts)
