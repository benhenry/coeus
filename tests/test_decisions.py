"""
Tests for the decision framework module.
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decisions import (
    DecisionFramework, DecisionRecord, DecisionType, DecisionStatus,
    format_decision_for_prompt
)


class TestRequiredCyclesCalculation:
    """Tests for the conviction cycle calculation."""

    @pytest.fixture
    def framework(self, temp_dir):
        """Create a DecisionFramework for testing."""
        return DecisionFramework(human_interaction_path=str(temp_dir))

    def test_high_confidence(self, framework):
        """99%+ confidence requires only 1 cycle."""
        assert framework.calculate_required_cycles(0.99) == 1
        assert framework.calculate_required_cycles(1.0) == 1

    def test_very_high_confidence(self, framework):
        """94-98% confidence requires 2 cycles."""
        assert framework.calculate_required_cycles(0.94) == 2
        assert framework.calculate_required_cycles(0.98) == 2

    def test_high_confidence_range(self, framework):
        """89-93% confidence requires 3 cycles."""
        assert framework.calculate_required_cycles(0.89) == 3
        assert framework.calculate_required_cycles(0.93) == 3

    def test_medium_confidence(self, framework):
        """84-88% confidence requires 4 cycles."""
        assert framework.calculate_required_cycles(0.84) == 4
        assert framework.calculate_required_cycles(0.88) == 4

    def test_lower_confidence(self, framework):
        """79-83% confidence requires 5 cycles."""
        assert framework.calculate_required_cycles(0.79) == 5
        assert framework.calculate_required_cycles(0.83) == 5

    def test_low_confidence(self, framework):
        """Below 79% returns -1 (gather more info)."""
        assert framework.calculate_required_cycles(0.78) == -1
        assert framework.calculate_required_cycles(0.5) == -1
        assert framework.calculate_required_cycles(0.0) == -1


class TestDecisionRecord:
    """Tests for the DecisionRecord dataclass."""

    def test_decision_creation(self):
        """Test basic decision creation."""
        decision = DecisionRecord(
            id="test_decision",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Test decision",
            reasoning="Test reasoning",
            counterarguments=["Counter 1"],
            confidence=0.9,
            created_cycle=1
        )
        assert decision.id == "test_decision"
        assert decision.status == DecisionStatus.CONSIDERING
        assert decision.conviction_cycles == 0

    def test_decision_to_dict(self):
        """Test serialization to dict."""
        decision = DecisionRecord(
            id="test_decision",
            decision_type=DecisionType.ONE_WAY_DOOR,
            summary="Important decision",
            reasoning="Careful reasoning",
            counterarguments=["Counterarg 1", "Counterarg 2"],
            confidence=0.95,
            created_cycle=5
        )
        d = decision.to_dict()
        assert d['id'] == "test_decision"
        assert d['decision_type'] == "one_way_door"
        assert d['status'] == "considering"
        assert len(d['counterarguments']) == 2

    def test_decision_from_dict(self):
        """Test deserialization from dict."""
        data = {
            'id': 'restored_decision',
            'decision_type': 'capability_request',
            'summary': 'Request new capability',
            'reasoning': 'Need it',
            'counterarguments': [],
            'confidence': 0.85,
            'confidence_history': [],
            'created_cycle': 10,
            'resolved_cycle': None,
            'status': 'pending_cycles',
            'conviction_cycles': 2,
            'required_cycles': 3,
            'human_response': None,
            'human_notes': None,
            'outcome': None,
            'outcome_matched_prediction': None
        }
        decision = DecisionRecord.from_dict(data)
        assert decision.id == 'restored_decision'
        assert decision.decision_type == DecisionType.CAPABILITY_REQUEST
        assert decision.status == DecisionStatus.PENDING_CYCLES
        assert decision.conviction_cycles == 2


class TestDecisionFramework:
    """Tests for the DecisionFramework class."""

    def test_create_decision(self, temp_dir):
        """Test creating a new decision."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="test_dec_1",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Simple decision",
            reasoning="It's straightforward",
            counterarguments=["Maybe not"],
            confidence=0.95,
            current_cycle=1
        )

        assert decision.id == "test_dec_1"
        assert decision.decision_type == DecisionType.TWO_WAY_DOOR
        assert decision.status == DecisionStatus.PENDING_CYCLES

    def test_one_way_door_needs_conviction_then_human(self, temp_dir):
        """Test that one-way door decisions need conviction cycles then human approval."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        # Create with 99% confidence (requires 1 cycle)
        decision = framework.create_decision(
            decision_id="one_way_1",
            decision_type=DecisionType.ONE_WAY_DOOR,
            summary="Irreversible decision",
            reasoning="This cannot be undone",
            counterarguments=[],
            confidence=0.99,
            current_cycle=1
        )

        # Initially PENDING_CYCLES
        assert decision.status == DecisionStatus.PENDING_CYCLES

        # After one update with consistent confidence, should go to AWAITING_HUMAN
        framework.update_decision_confidence(
            "one_way_1",
            new_confidence=0.99,
            current_cycle=2
        )

        updated = framework.active_decisions["one_way_1"]
        assert updated.status == DecisionStatus.AWAITING_HUMAN

    def test_capability_request_needs_conviction_then_human(self, temp_dir):
        """Test that capability requests need conviction then human approval."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="cap_req_1",
            decision_type=DecisionType.CAPABILITY_REQUEST,
            summary="Request web access",
            reasoning="Need to search the web",
            counterarguments=[],
            confidence=0.99,
            current_cycle=1
        )

        # Initially PENDING_CYCLES
        assert decision.status == DecisionStatus.PENDING_CYCLES

        # After conviction cycle, should go to AWAITING_HUMAN
        framework.update_decision_confidence(
            "cap_req_1",
            new_confidence=0.99,
            current_cycle=2
        )

        updated = framework.active_decisions["cap_req_1"]
        assert updated.status == DecisionStatus.AWAITING_HUMAN

    def test_update_confidence_increments_conviction(self, temp_dir):
        """Test that updating confidence increments conviction cycles."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="update_test",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Test decision",
            reasoning="Testing",
            counterarguments=[],
            confidence=0.90,  # Requires 3 cycles
            current_cycle=1
        )

        # Initial conviction is 1 (first cycle of confidence)
        initial_conviction = decision.conviction_cycles

        # Update with maintained high confidence
        framework.update_decision_confidence(
            "update_test",
            new_confidence=0.91,
            current_cycle=2,
            new_reasoning="Still confident"
        )

        updated = framework.active_decisions["update_test"]
        assert updated.conviction_cycles == initial_conviction + 1

    def test_confidence_drop_resets_conviction(self, temp_dir):
        """Test that significant confidence drop resets conviction cycles."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="drop_test",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Test decision",
            reasoning="Testing",
            counterarguments=[],
            confidence=0.95,
            current_cycle=1
        )

        framework.update_decision_confidence(
            "drop_test",
            new_confidence=0.96,
            current_cycle=2
        )

        # Significant confidence drop (more than 0.05)
        framework.update_decision_confidence(
            "drop_test",
            new_confidence=0.80,
            current_cycle=3
        )

        updated = framework.active_decisions["drop_test"]
        # Should reset to 1 (new confidence level starts a new conviction build)
        assert updated.conviction_cycles == 1

    def test_two_way_door_becomes_approved_after_cycles(self, temp_dir):
        """Test that two-way door decision becomes APPROVED after conviction cycles."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="ready_test",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Test decision",
            reasoning="Testing",
            counterarguments=[],
            confidence=0.99,  # Requires only 1 cycle
            current_cycle=1
        )

        # Initially PENDING_CYCLES
        assert decision.status == DecisionStatus.PENDING_CYCLES

        # After conviction cycle is met, TWO_WAY_DOOR should be APPROVED
        framework.update_decision_confidence(
            "ready_test",
            new_confidence=0.99,
            current_cycle=2
        )

        updated = framework.active_decisions["ready_test"]
        assert updated.status == DecisionStatus.APPROVED

    def test_get_pending_decisions(self, temp_dir):
        """Test getting pending decisions."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        framework.create_decision(
            decision_id="pending_1",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Decision 1",
            reasoning="Test",
            counterarguments=[],
            confidence=0.85,  # Requires 4 cycles
            current_cycle=1
        )
        framework.create_decision(
            decision_id="pending_2",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Decision 2",
            reasoning="Test",
            counterarguments=[],
            confidence=0.90,  # Requires 3 cycles
            current_cycle=1
        )

        pending = framework.get_pending_decisions()
        assert len(pending) == 2

    def test_mark_executed(self, temp_dir):
        """Test marking a decision as executed."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        decision = framework.create_decision(
            decision_id="exec_test",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Executable decision",
            reasoning="Test",
            counterarguments=[],
            confidence=0.99,
            current_cycle=1
        )

        # Build conviction to get APPROVED
        framework.update_decision_confidence("exec_test", 0.99, 2)

        framework.mark_executed("exec_test", "It worked!", True)

        updated = framework.active_decisions["exec_test"]
        assert updated.status == DecisionStatus.EXECUTED
        assert updated.outcome == "It worked!"
        assert updated.outcome_matched_prediction is True

    def test_process_human_response_approved(self, temp_dir):
        """Test processing human approval."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        framework.create_decision(
            decision_id="approval_test",
            decision_type=DecisionType.ONE_WAY_DOOR,
            summary="Needs approval",
            reasoning="Important",
            counterarguments=[],
            confidence=0.99,
            current_cycle=1
        )

        # Build conviction to get AWAITING_HUMAN
        framework.update_decision_confidence("approval_test", 0.99, 2)

        # Verify it's awaiting human
        assert framework.active_decisions["approval_test"].status == DecisionStatus.AWAITING_HUMAN

        framework.process_human_response(
            "approval_test",
            response="APPROVED",
            notes="Go ahead",
            current_cycle=3
        )

        updated = framework.active_decisions["approval_test"]
        assert updated.status == DecisionStatus.APPROVED
        assert updated.human_response == "APPROVED"

    def test_process_human_response_denied(self, temp_dir):
        """Test processing human denial."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        framework.create_decision(
            decision_id="deny_test",
            decision_type=DecisionType.ONE_WAY_DOOR,
            summary="Will be denied",
            reasoning="Risky",
            counterarguments=[],
            confidence=0.99,
            current_cycle=1
        )

        # Build conviction to get AWAITING_HUMAN
        framework.update_decision_confidence("deny_test", 0.99, 2)

        framework.process_human_response(
            "deny_test",
            response="DENIED",
            notes="Too risky",
            current_cycle=3
        )

        updated = framework.active_decisions["deny_test"]
        assert updated.status == DecisionStatus.DENIED

    def test_save_and_load_state(self, temp_dir):
        """Test state persistence."""
        framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )

        framework.create_decision(
            decision_id="persist_test",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Persistent decision",
            reasoning="Test persistence",
            counterarguments=["Maybe not"],
            confidence=0.90,
            current_cycle=1
        )

        state_path = temp_dir / "decisions.json"
        framework.save_state(str(state_path))

        # Load into new framework
        new_framework = DecisionFramework(
            human_interaction_path=str(temp_dir)
        )
        new_framework.load_state(str(state_path))

        loaded = new_framework.active_decisions.get("persist_test")
        assert loaded is not None
        assert loaded.summary == "Persistent decision"


class TestFormatDecisionForPrompt:
    """Tests for the format_decision_for_prompt function."""

    def test_format_basic_decision(self):
        """Test formatting a basic decision."""
        decision = DecisionRecord(
            id="test_format",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Format test",
            reasoning="Testing formatting",
            counterarguments=[],
            confidence=0.85,
            created_cycle=1
        )

        formatted = format_decision_for_prompt(decision)
        assert "Format test" in formatted
        assert "85" in formatted  # 85.0%
        assert "two_way_door" in formatted.lower()

    def test_format_decision_with_counterarguments(self):
        """Test formatting a decision with counterarguments."""
        decision = DecisionRecord(
            id="test_format",
            decision_type=DecisionType.TWO_WAY_DOOR,
            summary="Complex decision",
            reasoning="Detailed reasoning",
            counterarguments=["It might fail", "Alternative exists"],
            confidence=0.90,
            created_cycle=1
        )

        formatted = format_decision_for_prompt(decision)
        assert "It might fail" in formatted
        assert "Alternative exists" in formatted
