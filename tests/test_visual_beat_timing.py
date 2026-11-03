"""Tests for Stage 3.2: VisualBeat narration synchronization and beat timing assignment."""

import pytest
from app.subtitles.types import SubtitleResult, SubtitleSegment, WordTimestamp
from app.subtitles.alignment import (
    align_scene_visual_beats,
    apply_aligned_beat_timings_to_storyboard,
    compute_aligned_scene_timings,
)
from app.steps.storyboard import VisualBeat


def _make_word(text: str, start: float, end: float) -> WordTimestamp:
    return WordTimestamp(text=text, start=round(start, 3), end=round(end, 3), confidence=1.0)


def test_correct_mapping_of_beat_to_aligned_narration_words():
    """1. Correct mapping of a beat to aligned narration words."""
    # Narration: "Because the answer takes powerful computer chips,"
    words = [
        _make_word("Because", 0.10, 0.40),
        _make_word("the", 0.42, 0.55),
        _make_word("answer", 0.58, 0.95),
        _make_word("takes", 0.98, 1.25),
        _make_word("powerful", 1.28, 1.70),
        _make_word("computer", 1.75, 2.15),
        _make_word("chips,", 2.18, 2.60),
    ]

    beats = [
        VisualBeat(
            id="beat_1_1",
            scene_id="1",
            order=1,
            narration_text="Because the answer takes",
            visual_description="Establish question context",
        ),
        VisualBeat(
            id="beat_1_2",
            scene_id="1",
            order=2,
            narration_text="powerful computer chips,",
            visual_description="Close-up of computer chips",
        ),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=2.85,
        scene_narration="Because the answer takes powerful computer chips,",
    )

    assert len(aligned) == 2
    # Beat 1 maps to words 0..3 ("Because the answer takes")
    assert aligned[0]["start_word_index"] == 0
    assert aligned[0]["end_word_index"] == 3
    assert aligned[0]["order"] == 1

    # Beat 2 maps to words 4..6 ("powerful computer chips,")
    assert aligned[1]["start_word_index"] == 4
    assert aligned[1]["end_word_index"] == 6
    assert aligned[1]["order"] == 2


def test_repeated_words_and_phrases_at_multiple_positions():
    """2. Repeated words and phrases appearing at multiple positions are mapped to correct occurrences."""
    # Sentence has "chips" at word 6 and "chips" at word 9
    words = [
        _make_word("Because", 0.10, 0.40),      # 0
        _make_word("the", 0.42, 0.55),          # 1
        _make_word("answer", 0.58, 0.95),       # 2
        _make_word("takes", 0.98, 1.25),        # 3
        _make_word("powerful", 1.28, 1.70),     # 4
        _make_word("computer", 1.75, 2.15),     # 5
        _make_word("chips,", 2.18, 2.60),       # 6: first "chips"
        _make_word("and", 2.85, 3.05),          # 7
        _make_word("those", 3.08, 3.35),        # 8
        _make_word("chips", 3.38, 3.75),        # 9: second "chips"
        _make_word("use", 3.78, 4.02),          # 10
        _make_word("electricity", 4.05, 4.80),  # 11
        _make_word("every", 4.95, 5.20),        # 12
        _make_word("time", 5.22, 5.45),         # 13
        _make_word("they", 5.48, 5.65),         # 14
        _make_word("work.", 5.68, 6.10),        # 15
    ]

    beats = [
        VisualBeat(
            id="beat_1_1",
            scene_id="1",
            order=1,
            narration_text="Because the answer takes powerful computer chips,",
            visual_description="Wide shot of motherboard with processor",
        ),
        VisualBeat(
            id="beat_1_2",
            scene_id="1",
            order=2,
            narration_text="and those chips use electricity",
            visual_description="Close-up of processor core die drawing power",
        ),
        VisualBeat(
            id="beat_1_3",
            scene_id="1",
            order=3,
            narration_text="every time they work.",
            visual_description="Macro shot of glowing electrical traces",
        ),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=6.35,
        scene_narration="Because the answer takes powerful computer chips, and those chips use electricity every time they work.",
    )

    assert len(aligned) == 3
    # Beat 1 contains the FIRST "chips," (index 6)
    assert aligned[0]["start_word_index"] == 0
    assert aligned[0]["end_word_index"] == 6
    assert words[aligned[0]["end_word_index"]].text == "chips,"

    # Beat 2 contains the SECOND "chips" (index 9)
    assert aligned[1]["start_word_index"] == 7
    assert aligned[1]["end_word_index"] == 11
    assert words[aligned[1]["start_word_index"]].text == "and"
    assert words[aligned[1]["end_word_index"]].text == "electricity"

    # Beat 3 contains the remainder
    assert aligned[2]["start_word_index"] == 12
    assert aligned[2]["end_word_index"] == 15


def test_correct_timing_derived_from_aligned_word_timestamps():
    """3. Correct timing derived from aligned word timestamps with pause snapping."""
    words = [
        _make_word("Why", 0.15, 0.45),          # 0
        _make_word("does", 0.48, 0.70),         # 1
        _make_word("one", 0.72, 0.95),          # 2
        _make_word("question", 0.98, 1.50),     # 3 -- gap between 1.50 and 1.80 = 0.30s
        _make_word("cost", 1.80, 2.15),         # 4
        _make_word("so", 2.18, 2.40),           # 5
        _make_word("much?", 2.42, 2.90),        # 6
    ]

    beats = [
        VisualBeat(
            id="b1",
            scene_id="1",
            order=1,
            narration_text="Why does one question",
            visual_description="Hook question",
        ),
        VisualBeat(
            id="b2",
            scene_id="1",
            order=2,
            narration_text="cost so much?",
            visual_description="Cost graph",
        ),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=3.10,
    )

    assert len(aligned) == 2
    # Beat 1 starts at scene start
    assert aligned[0]["start_time"] == 0.0

    # Cut between word 3 (ends at 1.50) and word 4 (starts at 1.80).
    # Gap is 0.30s. Cut lands inside pause: 1.50 + min(0.30 * 0.55, 0.30 - 0.04) = 1.50 + 0.165 = 1.665s.
    expected_cut = round(1.50 + min(0.30 * 0.55, 0.30 - 0.04), 3)
    assert aligned[0]["end_time"] == expected_cut
    assert aligned[1]["start_time"] == expected_cut
    assert aligned[1]["end_time"] == 3.10


def test_ordered_beat_timing_and_non_overlapping_intervals():
    """4 & 5. Ordered beat timing with non-overlapping, strictly gapless intervals."""
    words = [
        _make_word("Inside", 0.10, 0.40),
        _make_word("modern", 0.42, 0.80),
        _make_word("data", 0.82, 1.10),
        _make_word("centers,", 1.12, 1.60),
        _make_word("rows", 1.85, 2.15),
        _make_word("of", 2.16, 2.30),
        _make_word("high-density", 2.32, 2.90),
        _make_word("server", 2.92, 3.25),
        _make_word("racks", 3.28, 3.65),
        _make_word("operate", 3.75, 4.15),
        _make_word("continuously.", 4.18, 4.85),
    ]

    beats = [
        VisualBeat(id="b1", scene_id="1", order=1, narration_text="Inside modern data centers,", visual_description="Aisle"),
        VisualBeat(id="b2", scene_id="1", order=2, narration_text="rows of high-density server racks", visual_description="Racks"),
        VisualBeat(id="b3", scene_id="1", order=3, narration_text="operate continuously.", visual_description="Lights"),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=5.20,
    )

    assert len(aligned) == 3

    # Check strict ordering and positive duration
    for b in aligned:
        assert b["end_time"] > b["start_time"]
        assert b["duration_seconds"] == pytest.approx(b["end_time"] - b["start_time"], abs=0.002)

    # Check non-overlapping and gapless boundaries
    assert aligned[0]["start_time"] == 0.0
    assert aligned[0]["end_time"] == aligned[1]["start_time"]
    assert aligned[1]["end_time"] == aligned[2]["start_time"]
    assert aligned[2]["end_time"] == 5.20


def test_empty_or_missing_beat_lists():
    """6. Empty or missing beat lists handled safely without failure."""
    words = [_make_word("Aircraft", 0.1, 0.6), _make_word("windows.", 0.65, 1.2)]

    # Empty list of beats returns empty list
    aligned = align_scene_visual_beats(
        scene_beats=[],
        words=words,
        scene_start_time=0.0,
        scene_end_time=1.5,
    )
    assert aligned == []


def test_unmatched_or_ambiguous_narration_phrases_fallback():
    """7. Unmatched or ambiguous narration phrases fall back to safe proportional partition."""
    words = [
        _make_word("first", 0.2, 0.6),
        _make_word("second", 0.65, 1.1),
        _make_word("third", 1.15, 1.6),
        _make_word("fourth", 1.65, 2.1),
    ]

    # Beat phrases have completely hallucinated or mismatched words
    beats = [
        VisualBeat(id="b1", scene_id="1", order=1, narration_text="completely unrelated phrase", visual_description="desc1"),
        VisualBeat(id="b2", scene_id="1", order=2, narration_text="another alien sentence", visual_description="desc2"),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=2.4,
    )

    assert len(aligned) == 2
    # Still valid, ordered, positive duration
    assert aligned[0]["end_time"] > aligned[0]["start_time"]
    assert aligned[1]["end_time"] > aligned[1]["start_time"]
    assert aligned[0]["end_time"] == aligned[1]["start_time"]
    assert aligned[0]["start_time"] == 0.0
    assert aligned[1]["end_time"] == 2.4


def test_punctuation_and_continuous_speech_boundary_handling():
    """8. Punctuation and continuous speech boundaries cut cleanly at phonetic bounds."""
    # Words touch acoustically with zero gap (continuous fast speech)
    words = [
        _make_word("red", 0.10, 0.45),
        _make_word("hot", 0.45, 0.80),    # touches red exactly at 0.45
        _make_word("metal.", 0.80, 1.25),  # touches hot exactly at 0.80
    ]

    beats = [
        VisualBeat(id="b1", scene_id="1", order=1, narration_text="red", visual_description="Red glow"),
        VisualBeat(id="b2", scene_id="1", order=2, narration_text="hot metal.", visual_description="Metal structure"),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=1.40,
    )

    assert len(aligned) == 2
    # Cut occurs exactly where word 0 ends (0.45s) without slicing inside the word
    assert aligned[0]["end_time"] == 0.45
    assert aligned[1]["start_time"] == 0.45


def test_legacy_scenes_without_visual_beats():
    """9. Legacy scenes without VisualBeats remain 100% compatible."""
    scenes = [
        {"index": 1, "narration": "First scene narration."},
        {"index": 2, "narration": "Second scene narration."},
    ]

    seg1 = SubtitleSegment(
        index=1,
        text="First scene narration.",
        start=0.1,
        end=1.9,
        words=[_make_word("First", 0.1, 0.5), _make_word("scene", 0.52, 0.9), _make_word("narration.", 0.95, 1.8)],
    )
    seg2 = SubtitleSegment(
        index=2,
        text="Second scene narration.",
        start=2.2,
        end=3.8,
        words=[_make_word("Second", 2.2, 2.6), _make_word("scene", 2.62, 3.0), _make_word("narration.", 3.05, 3.7)],
    )

    alignment = SubtitleResult(
        segments=[seg1, seg2],
        total_words=6,
        total_duration_seconds=4.0,
        provider="mock",
        model="mock",
    )

    timings = compute_aligned_scene_timings(scenes, alignment, 4.0)
    assert len(timings) == 2
    assert timings[0]["start_time"] == 0.0
    assert timings[1]["end_time"] == 4.0
    # No visual_beats key in timings for legacy scenes that didn't provide visual_beats
    assert "visual_beats" not in timings[0]
    assert "visual_beats" not in timings[1]


def test_storyboard_with_visual_beats_receives_aligned_timestamps():
    """10. Scenes with visual_beats receive aligned beat timestamps in compute_aligned_scene_timings."""
    scenes = [
        {
            "index": 1,
            "narration": "Air travelers notice a tiny bleed hole.",
            "visual_beats": [
                {
                    "id": "beat_1_1",
                    "order": 1,
                    "narration_text": "Air travelers notice",
                    "visual_description": "Passenger cabin",
                },
                {
                    "id": "beat_1_2",
                    "order": 2,
                    "narration_text": "a tiny bleed hole.",
                    "visual_description": "Close-up on bleed hole",
                },
            ],
        }
    ]

    seg = SubtitleSegment(
        index=1,
        text="Air travelers notice a tiny bleed hole.",
        start=0.10,
        end=2.80,
        words=[
            _make_word("Air", 0.10, 0.35),
            _make_word("travelers", 0.38, 0.90),
            _make_word("notice", 0.92, 1.30),
            _make_word("a", 1.45, 1.55),
            _make_word("tiny", 1.58, 1.95),
            _make_word("bleed", 1.98, 2.30),
            _make_word("hole.", 2.32, 2.75),
        ],
    )

    alignment = SubtitleResult(
        segments=[seg],
        total_words=7,
        total_duration_seconds=3.0,
        provider="mock",
        model="mock",
    )

    timings = compute_aligned_scene_timings(scenes, alignment, 3.0)
    assert len(timings) == 1
    assert "visual_beats" in timings[0]
    beats = timings[0]["visual_beats"]
    assert len(beats) == 2

    # Beat 1
    assert beats[0]["start_word_index"] == 0
    assert beats[0]["end_word_index"] == 2
    assert beats[0]["start_time"] == 0.0
    assert beats[0]["end_time"] > beats[0]["start_time"]
    assert beats[0]["timing_type"] == "aligned"

    # Beat 2
    assert beats[1]["start_word_index"] == 3
    assert beats[1]["end_word_index"] == 6
    assert beats[1]["start_time"] == beats[0]["end_time"]
    assert beats[1]["end_time"] == 3.0


def test_apply_aligned_beat_timings_to_storyboard_helper():
    """11. apply_aligned_beat_timings_to_storyboard updates storyboard payload in-place."""
    storyboard = {
        "storyboard": {
            "scenes": [
                {
                    "index": 1,
                    "narration": "First beat and second beat.",
                    "visual_beats": [
                        {"id": "b1", "order": 1, "narration_text": "First beat", "visual_description": "d1"},
                        {"id": "b2", "order": 2, "narration_text": "and second beat.", "visual_description": "d2"},
                    ],
                }
            ]
        }
    }

    seg = SubtitleSegment(
        index=1,
        text="First beat and second beat.",
        start=0.1,
        end=2.0,
        words=[
            _make_word("First", 0.1, 0.4),
            _make_word("beat", 0.42, 0.8),
            _make_word("and", 0.95, 1.1),
            _make_word("second", 1.12, 1.5),
            _make_word("beat.", 1.52, 1.95),
        ],
    )

    alignment = SubtitleResult(
        segments=[seg],
        total_words=5,
        total_duration_seconds=2.2,
        provider="mock",
        model="mock",
    )

    updated = apply_aligned_beat_timings_to_storyboard(storyboard, alignment, 2.2)
    sc = updated["storyboard"]["scenes"][0]
    assert sc["start_time"] == 0.0
    assert sc["end_time"] == 2.2
    assert len(sc["visual_beats"]) == 2
    assert sc["visual_beats"][0]["start_time"] == 0.0
    assert sc["visual_beats"][0]["end_time"] == sc["visual_beats"][1]["start_time"]
    assert sc["visual_beats"][1]["end_time"] == 2.2


def test_more_beats_than_words_fallback():
    """12. Edge case where scene has fewer words than beats."""
    words = [_make_word("Single", 0.1, 0.5)]

    beats = [
        VisualBeat(id="b1", scene_id="1", order=1, narration_text="Single", visual_description="v1"),
        VisualBeat(id="b2", scene_id="1", order=2, narration_text="", visual_description="v2"),
    ]

    aligned = align_scene_visual_beats(
        scene_beats=beats,
        words=words,
        scene_start_time=0.0,
        scene_end_time=1.0,
    )

    assert len(aligned) == 2
    assert aligned[0]["end_time"] > aligned[0]["start_time"]
    assert aligned[1]["end_time"] > aligned[1]["start_time"]
    assert aligned[0]["end_time"] == aligned[1]["start_time"]
    assert aligned[1]["end_time"] == 1.0

