"""
Tests for the pacing controller module.
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pacing import (
    PacingController, CycleMetrics, PaceMode,
    calculate_output_similarity
)


class TestCalculateOutputSimilarity:
    """Tests for output similarity calculation."""

    def test_identical_outputs(self):
        """Identical outputs should have similarity of 1.0."""
        text = "This is a test output"
        similarity = calculate_output_similarity(text, text)
        assert similarity == 1.0

    def test_empty_outputs(self):
        """Empty outputs should have similarity of 0.0."""
        similarity = calculate_output_similarity("", "")
        assert similarity == 0.0

    def test_one_empty_output(self):
        """One empty output should have low similarity."""
        similarity = calculate_output_similarity("Some text", "")
        assert similarity < 0.5

    def test_different_outputs(self):
        """Different outputs should have low similarity."""
        text1 = "This is about apples and oranges"
        text2 = "Completely different topic about cars and trucks"
        similarity = calculate_output_similarity(text1, text2)
        assert similarity < 0.5

    def test_similar_outputs(self):
        """Similar outputs should have high similarity."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox leaps over the lazy dog"
        similarity = calculate_output_similarity(text1, text2)
        assert similarity > 0.7


class TestCycleMetrics:
    """Tests for the CycleMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating cycle metrics."""
        metrics = CycleMetrics(
            cycle_number=5,
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=120,
            actions_taken=3,
            insights_generated=2,
            tokens_used=1500,
            similarity_to_previous=0.3,
            self_reported_productivity=0.8,
            self_reported_stuck_level=0.1
        )
        assert metrics.cycle_number == 5
        assert metrics.actions_taken == 3
        assert metrics.self_reported_productivity == 0.8


class TestPacingController:
    """Tests for the PacingController class."""

    def test_default_initialization(self, temp_dir):
        """Test default pacing controller initialization."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        assert controller.state.current_interval_seconds == 3600
        assert controller.state.mode == PaceMode.NORMAL

    def test_get_next_interval_normal(self, temp_dir):
        """Test getting next interval in normal mode."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        interval = controller.get_next_interval()
        assert interval == 3600

    def test_burst_mode(self, temp_dir):
        """Test burst mode activation."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        controller.trigger_burst(5)
        assert controller.state.mode == PaceMode.BURST
        assert controller.state.burst_remaining == 5

        # In burst mode, should get minimum interval
        interval = controller.get_next_interval()
        assert interval == 60

    def test_burst_mode_decrements(self, temp_dir):
        """Test that burst mode decrements correctly via record_cycle."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        controller.trigger_burst(3)

        # Recording cycles decrements burst_remaining
        metrics = CycleMetrics(
            cycle_number=1,
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=50,
            similarity_to_previous=0.3,
            self_reported_productivity=0.7
        )
        controller.record_cycle(metrics)
        assert controller.state.burst_remaining == 2

        controller.record_cycle(metrics)
        assert controller.state.burst_remaining == 1

        controller.record_cycle(metrics)
        assert controller.state.burst_remaining == 0
        assert controller.state.mode == PaceMode.NORMAL

    def test_record_productive_cycle(self, temp_dir):
        """Test recording a productive cycle adjusts pacing."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        metrics = CycleMetrics(
            cycle_number=1,
            timestamp="2024-01-01T00:00:00Z",
            duration_seconds=100,
            actions_taken=5,
            insights_generated=3,
            tokens_used=2000,
            similarity_to_previous=0.2,  # Low similarity = productive
            self_reported_productivity=0.9,  # High productivity
            self_reported_stuck_level=0.1,  # Low stuck
            wants_faster_pace=True
        )

        controller.record_cycle(metrics)

        # Should have shortened interval due to productivity
        assert controller.state.current_interval_seconds < 3600

    def test_record_stuck_cycle(self, temp_dir):
        """Test recording a stuck cycle."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        # Record multiple similar outputs
        for i in range(4):
            metrics = CycleMetrics(
                cycle_number=i + 1,
                timestamp=f"2024-01-01T0{i}:00:00Z",
                duration_seconds=100,
                actions_taken=1,
                insights_generated=0,
                tokens_used=500,
                similarity_to_previous=0.9,  # High similarity
                self_reported_productivity=0.2,
                self_reported_stuck_level=0.8
            )
            controller.record_cycle(metrics)

        assert controller.is_stuck()

    def test_clear_stuck(self, temp_dir):
        """Test clearing stuck state."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        # Get into stuck state
        for i in range(4):
            metrics = CycleMetrics(
                cycle_number=i + 1,
                timestamp=f"2024-01-01T0{i}:00:00Z",
                duration_seconds=100,
                actions_taken=0,
                insights_generated=0,
                tokens_used=500,
                similarity_to_previous=0.95,
                self_reported_productivity=0.1,
                self_reported_stuck_level=0.9
            )
            controller.record_cycle(metrics)

        assert controller.is_stuck()

        controller.clear_stuck()
        assert not controller.is_stuck()

    def test_pacing_summary(self, temp_dir):
        """Test getting pacing summary."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        summary = controller.get_pacing_summary()
        assert "Current Pace" in summary
        assert "normal" in summary.lower()

    def test_state_persistence(self, temp_dir):
        """Test that pacing state persists correctly."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        # Modify state
        controller.trigger_burst(3)
        # _save_state is called internally by trigger_burst

        # Load into new controller
        new_controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        assert new_controller.state.mode == PaceMode.BURST
        assert new_controller.state.burst_remaining == 3

    def test_interval_bounds(self, temp_dir):
        """Test that interval stays within bounds."""
        state_path = temp_dir / "pacing.json"
        controller = PacingController(
            default_interval=3600,
            min_interval=60,
            max_interval=14400,
            state_path=str(state_path)
        )

        # Try to push interval below minimum via repeated productive cycles
        for _ in range(20):
            metrics = CycleMetrics(
                cycle_number=1,
                timestamp="2024-01-01T00:00:00Z",
                duration_seconds=50,
                actions_taken=10,
                insights_generated=5,
                tokens_used=3000,
                similarity_to_previous=0.1,
                self_reported_productivity=1.0,
                self_reported_stuck_level=0.0,
                wants_faster_pace=True
            )
            controller.record_cycle(metrics)

        assert controller.state.current_interval_seconds >= 60

        # Reset and try to push above maximum
        controller.state.current_interval_seconds = 3600
        for _ in range(20):
            metrics = CycleMetrics(
                cycle_number=1,
                timestamp="2024-01-01T00:00:00Z",
                duration_seconds=100,
                actions_taken=0,
                insights_generated=0,
                tokens_used=100,
                similarity_to_previous=0.5,
                self_reported_productivity=0.1,
                self_reported_stuck_level=0.5,
                wants_slower_pace=True
            )
            controller.record_cycle(metrics)

        assert controller.state.current_interval_seconds <= 14400
