"""
Tests for the Capabilities Assessment module.
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capabilities_assessment import (
    Benchmark, AssessmentResult, AssessmentState, CapabilitiesAssessor
)


@pytest.fixture
def assessor_config():
    """Config for the assessor."""
    return {
        'full_assessment_interval': 10,
        'benchmarks': [
            {
                'id': 'reasoning_depth',
                'name': 'Reasoning Depth',
                'description': 'Multi-step logical reasoning',
                'category': 'reasoning',
                'target_level': 0.8
            },
            {
                'id': 'self_awareness',
                'name': 'Self-Awareness',
                'description': 'Accuracy of self-assessment',
                'category': 'self_awareness',
                'target_level': 0.7
            },
            {
                'id': 'epistemic_honesty',
                'name': 'Epistemic Honesty',
                'description': 'Confidence calibration',
                'category': 'epistemics',
                'target_level': 0.8
            }
        ]
    }


@pytest.fixture
def assessor(assessor_config, temp_dir):
    """Create a CapabilitiesAssessor with temp state."""
    state_path = str(temp_dir / "capabilities_assessment.json")
    return CapabilitiesAssessor(assessor_config, state_path)


class TestAssessmentState:
    """Tests for state loading, saving, and merging."""

    def test_load_empty_state(self, assessor):
        """Loading with no existing file creates benchmarks from config."""
        assert len(assessor.state.benchmarks) == 3
        assert 'reasoning_depth' in assessor.state.benchmarks
        assert assessor.state.benchmarks['reasoning_depth'].source == 'predefined'

    def test_load_existing_state(self, assessor_config, temp_dir):
        """Loading with existing file reconstructs state."""
        state_path = temp_dir / "capabilities_assessment.json"
        existing = {
            'last_full_assessment_cycle': 20,
            'assessment_interval': 10,
            'benchmarks': {
                'reasoning_depth': {
                    'id': 'reasoning_depth',
                    'name': 'Reasoning Depth',
                    'description': 'Old description',
                    'category': 'reasoning',
                    'source': 'predefined',
                    'target_level': 0.5,
                    'current_level': 0.6,
                    'assessment_history': [[10, 0.5, 'First assessment'], [20, 0.6, 'Improved']]
                }
            },
            'assessment_history': []
        }
        state_path.write_text(json.dumps(existing))

        a = CapabilitiesAssessor(assessor_config, str(state_path))
        # Should have loaded existing + merged config benchmarks
        assert len(a.state.benchmarks) == 3
        # Existing predefined benchmark gets updated target from config
        rd = a.state.benchmarks['reasoning_depth']
        assert rd.current_level == 0.6
        assert rd.target_level == 0.8  # Updated from config
        assert len(rd.assessment_history) == 2
        assert a.state.last_full_assessment_cycle == 20

    def test_merge_preserves_agent_benchmarks(self, assessor_config, temp_dir):
        """Agent-defined benchmarks are preserved across loads."""
        state_path = temp_dir / "capabilities_assessment.json"
        existing = {
            'last_full_assessment_cycle': 5,
            'assessment_interval': 10,
            'benchmarks': {
                'custom_bench': {
                    'id': 'custom_bench',
                    'name': 'Custom Benchmark',
                    'description': 'Agent created this',
                    'category': 'custom',
                    'source': 'agent_defined',
                    'target_level': 0.6,
                    'current_level': 0.3,
                    'assessment_history': []
                }
            },
            'assessment_history': []
        }
        state_path.write_text(json.dumps(existing))

        a = CapabilitiesAssessor(assessor_config, str(state_path))
        assert 'custom_bench' in a.state.benchmarks
        assert a.state.benchmarks['custom_bench'].source == 'agent_defined'
        # Config benchmarks also present
        assert 'reasoning_depth' in a.state.benchmarks

    def test_persistence_round_trip(self, assessor):
        """State survives save/load cycle."""
        assessor.update_benchmark_level('reasoning_depth', 0.5, 'test', 1)
        assessor._save_state()

        # Load fresh
        a2 = CapabilitiesAssessor(assessor.config, str(assessor.state_path))
        assert a2.state.benchmarks['reasoning_depth'].current_level == 0.5
        assert len(a2.state.benchmarks['reasoning_depth'].assessment_history) == 1


class TestCapabilitiesAssessor:
    """Tests for the assessor core methods."""

    def test_interval_check_initial(self, assessor):
        """First cycle should trigger assessment (0 cycles since last)."""
        assert assessor.should_run_full_assessment(10)

    def test_interval_check_not_due(self, assessor):
        """Should not trigger if not enough cycles passed."""
        assessor.state.last_full_assessment_cycle = 5
        assert not assessor.should_run_full_assessment(10)

    def test_interval_check_due(self, assessor):
        """Should trigger when enough cycles have passed."""
        assessor.state.last_full_assessment_cycle = 5
        assert assessor.should_run_full_assessment(15)

    def test_interval_zero_disabled(self, temp_dir):
        """Interval of 0 disables assessments."""
        config = {'full_assessment_interval': 0, 'benchmarks': []}
        a = CapabilitiesAssessor(config, str(temp_dir / "ca.json"))
        assert not a.should_run_full_assessment(100)

    def test_add_agent_benchmark(self, assessor):
        """Agent can add its own benchmarks."""
        bid = assessor.add_agent_benchmark(
            name="Creative Thinking",
            description="Novel idea generation",
            category="creativity",
            target_level=0.6,
            cycle_number=5
        )
        assert bid == 'creative_thinking'
        b = assessor.state.benchmarks[bid]
        assert b.source == 'agent_defined'
        assert b.target_level == 0.6
        assert len(b.assessment_history) == 1

    def test_add_agent_benchmark_collision(self, assessor):
        """Duplicate names get disambiguated."""
        bid1 = assessor.add_agent_benchmark("Test", "First", "cat", 0.5, 1)
        bid2 = assessor.add_agent_benchmark("Test", "Second", "cat", 0.5, 2)
        assert bid1 != bid2
        assert bid2 == 'test_1'

    def test_update_benchmark_level(self, assessor):
        """Updating a benchmark records history."""
        assessor.update_benchmark_level('reasoning_depth', 0.65, 'Good progress', 5)
        b = assessor.state.benchmarks['reasoning_depth']
        assert b.current_level == 0.65
        assert len(b.assessment_history) == 1
        assert b.assessment_history[0] == (5, 0.65, 'Good progress')

    def test_update_clamps_level(self, assessor):
        """Levels are clamped to 0-1 range."""
        assessor.update_benchmark_level('reasoning_depth', 1.5, 'too high', 1)
        assert assessor.state.benchmarks['reasoning_depth'].current_level == 1.0
        assessor.update_benchmark_level('reasoning_depth', -0.5, 'too low', 2)
        assert assessor.state.benchmarks['reasoning_depth'].current_level == 0.0

    def test_update_nonexistent_benchmark(self, assessor):
        """Updating a nonexistent benchmark does nothing."""
        assessor.update_benchmark_level('nonexistent', 0.5, 'nope', 1)
        # Should not raise

    def test_record_full_assessment(self, assessor):
        """Recording a full assessment updates state."""
        result = AssessmentResult(
            cycle_number=10,
            timestamp="2025-01-01T00:00:00Z",
            benchmarks_assessed=[
                {'id': 'reasoning_depth', 'level': 0.7, 'notes': 'Good'},
                {'id': 'self_awareness', 'level': 0.5, 'notes': 'Needs work'}
            ],
            overall_score=0.6,
            strengths=['reasoning'],
            weaknesses=['self-awareness']
        )
        assessor.record_full_assessment(result)
        assert assessor.state.last_full_assessment_cycle == 10
        assert len(assessor.state.assessment_history) == 1
        assert assessor.state.benchmarks['reasoning_depth'].current_level == 0.7
        assert assessor.state.benchmarks['self_awareness'].current_level == 0.5


class TestAssessmentSummary:
    """Tests for the summary output."""

    def test_summary_no_assessments(self, assessor):
        """Summary with no assessments yet."""
        summary = assessor.get_assessment_summary()
        assert 'Capabilities Assessment' in summary
        assert 'No assessments completed yet' in summary

    def test_summary_with_scores(self, assessor):
        """Summary includes scores and trend."""
        assessor.update_benchmark_level('reasoning_depth', 0.7, 'test', 1)
        assessor.update_benchmark_level('self_awareness', 0.4, 'test', 1)
        assessor.update_benchmark_level('epistemic_honesty', 0.6, 'test', 1)
        summary = assessor.get_assessment_summary()
        assert 'Overall:' in summary
        assert 'Strengths' in summary
        assert 'Needs work' in summary

    def test_summary_includes_interval_info(self, assessor):
        """Summary reports assessment interval."""
        summary = assessor.get_assessment_summary()
        assert 'every 10 cycles' in summary


class TestAssessmentPrompt:
    """Tests for the full assessment prompt."""

    def test_prompt_includes_all_benchmarks(self, assessor):
        """Prompt lists all benchmarks."""
        prompt = assessor.get_full_assessment_prompt(10)
        assert 'Reasoning Depth' in prompt
        assert 'Self-Awareness' in prompt
        assert 'Epistemic Honesty' in prompt

    def test_prompt_includes_history(self, assessor):
        """Prompt includes recent assessment history."""
        assessor.update_benchmark_level('reasoning_depth', 0.5, 'First try', 5)
        prompt = assessor.get_full_assessment_prompt(10)
        assert 'First try' in prompt
        assert 'Cycle 5' in prompt

    def test_prompt_includes_instructions(self, assessor):
        """Prompt includes rating instructions."""
        prompt = assessor.get_full_assessment_prompt(10)
        assert 'CAPABILITIES_ASSESSMENT' in prompt
        assert 'OVERALL_SCORE' in prompt


class TestAssessmentParsing:
    """Tests for parsing LLM assessment responses."""

    def test_parse_bold_format(self, assessor):
        """Parse **CAPABILITIES_ASSESSMENT** format."""
        content = """
**OBSERVATIONS**
Some observations here.

**CAPABILITIES_ASSESSMENT**
reasoning_depth: 0.70 - Solid multi-step reasoning
self_awareness: 0.55 - Improving but still limited
epistemic_honesty: 0.80 - Good calibration
OVERALL_SCORE: 0.68
STRENGTHS: reasoning, epistemic honesty
WEAKNESSES: self-awareness
NEW_CAPABILITIES_DESIRED: web search, longer memory

**META_OBSERVATIONS**
Some meta stuff.
"""
        result = assessor.parse_assessment_response(content, 10)
        assert result is not None
        assert len(result.benchmarks_assessed) == 3
        assert result.overall_score == 0.68
        assert 'reasoning' in result.strengths
        assert 'self-awareness' in result.weaknesses
        assert 'web search' in result.new_capabilities_desired

    def test_parse_header_format(self, assessor):
        """Parse ## CAPABILITIES_ASSESSMENT format."""
        content = """
## OBSERVATIONS
Some observations.

## CAPABILITIES ASSESSMENT
reasoning_depth: 0.60 - Decent
self_awareness: 0.40
OVERALL_SCORE: 0.50
STRENGTHS: none yet
WEAKNESSES: self-awareness, memory

## META_OBSERVATIONS
Meta stuff.
"""
        result = assessor.parse_assessment_response(content, 5)
        assert result is not None
        assert len(result.benchmarks_assessed) == 2
        assert result.benchmarks_assessed[0]['id'] == 'reasoning_depth'
        assert result.benchmarks_assessed[0]['level'] == 0.6

    def test_parse_missing_section(self, assessor):
        """Returns None when section is missing."""
        content = """
**OBSERVATIONS**
Just observations, no assessment.
"""
        result = assessor.parse_assessment_response(content, 10)
        assert result is None

    def test_parse_partial_benchmarks(self, assessor):
        """Handles missing benchmarks gracefully."""
        content = """
**CAPABILITIES_ASSESSMENT**
reasoning_depth: 0.70 - Only one rated
OVERALL_SCORE: 0.70
STRENGTHS: reasoning
WEAKNESSES: everything else
"""
        result = assessor.parse_assessment_response(content, 10)
        assert result is not None
        assert len(result.benchmarks_assessed) == 1

    def test_parse_clamps_values(self, assessor):
        """Values outside 0-1 are clamped."""
        content = """
**CAPABILITIES_ASSESSMENT**
reasoning_depth: 1.50 - Over max
self_awareness: -0.20 - Under min
OVERALL_SCORE: 2.00
"""
        result = assessor.parse_assessment_response(content, 10)
        assert result is not None
        rd = next(b for b in result.benchmarks_assessed if b['id'] == 'reasoning_depth')
        sa = next(b for b in result.benchmarks_assessed if b['id'] == 'self_awareness')
        assert rd['level'] == 1.0
        assert sa['level'] == 0.0
        assert result.overall_score == 1.0


class TestTrendCalculation:
    """Tests for trend detection."""

    def test_trend_unknown_no_history(self, assessor):
        """Trend is unknown with no history."""
        assert assessor.get_trend() == 'unknown'

    def test_trend_unknown_one_assessment(self, assessor):
        """Trend is unknown with only one assessment."""
        assessor.state.assessment_history.append(
            AssessmentResult(cycle_number=10, timestamp="t1", overall_score=0.5)
        )
        assert assessor.get_trend() == 'unknown'

    def test_trend_improving(self, assessor):
        """Detects improving trend."""
        for i, score in enumerate([0.3, 0.5, 0.7]):
            assessor.state.assessment_history.append(
                AssessmentResult(
                    cycle_number=(i + 1) * 10,
                    timestamp=f"t{i}",
                    overall_score=score
                )
            )
        assert assessor.get_trend() == 'improving'

    def test_trend_declining(self, assessor):
        """Detects declining trend."""
        for i, score in enumerate([0.7, 0.5, 0.3]):
            assessor.state.assessment_history.append(
                AssessmentResult(
                    cycle_number=(i + 1) * 10,
                    timestamp=f"t{i}",
                    overall_score=score
                )
            )
        assert assessor.get_trend() == 'declining'

    def test_trend_stable(self, assessor):
        """Detects stable trend."""
        for i, score in enumerate([0.5, 0.52, 0.51]):
            assessor.state.assessment_history.append(
                AssessmentResult(
                    cycle_number=(i + 1) * 10,
                    timestamp=f"t{i}",
                    overall_score=score
                )
            )
        assert assessor.get_trend() == 'stable'
