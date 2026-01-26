"""
Tests for the goal management module.
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from goals import Goal, GoalTree, GoalStatus, GoalPriority, format_goals_for_prompt


class TestGoal:
    """Tests for the Goal dataclass."""

    def test_goal_creation(self):
        """Test basic goal creation."""
        goal = Goal(
            id="test_goal",
            content="Test content",
            reasoning="Test reasoning"
        )
        assert goal.id == "test_goal"
        assert goal.content == "Test content"
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority == GoalPriority.NORMAL
        assert goal.is_root is False

    def test_goal_to_dict(self):
        """Test Goal serialization to dict."""
        goal = Goal(
            id="test_goal",
            content="Test content",
            reasoning="Test reasoning",
            priority=GoalPriority.HIGH
        )
        d = goal.to_dict()
        assert d['id'] == "test_goal"
        assert d['content'] == "Test content"
        assert d['status'] == "active"
        assert d['priority'] == "high"

    def test_goal_from_dict(self):
        """Test Goal deserialization from dict."""
        data = {
            'id': 'test_goal',
            'content': 'Test content',
            'reasoning': 'Test reasoning',
            'parent_id': None,
            'children_ids': [],
            'status': 'completed',
            'priority': 'critical',
            'progress_notes': [(1, 'Note')],
            'completion_criteria': 'Done when done',
            'created_cycle': 5,
            'completed_cycle': 10,
            'is_root': False
        }
        goal = Goal.from_dict(data)
        assert goal.id == 'test_goal'
        assert goal.status == GoalStatus.COMPLETED
        assert goal.priority == GoalPriority.CRITICAL
        assert goal.completed_cycle == 10

    def test_goal_roundtrip(self):
        """Test serialization roundtrip."""
        original = Goal(
            id="roundtrip_test",
            content="Roundtrip content",
            reasoning="Roundtrip reasoning",
            priority=GoalPriority.HIGH,
            progress_notes=[(1, "First note"), (2, "Second note")]
        )
        serialized = original.to_dict()
        restored = Goal.from_dict(serialized)
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.priority == original.priority
        assert len(restored.progress_notes) == 2


class TestGoalTree:
    """Tests for the GoalTree class."""

    def test_create_new_goal_tree(self, temp_dir):
        """Test creating a new goal tree from scratch."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Test root goal",
            state_path=str(state_path)
        )

        assert tree.root_goal_id == "root_goal"
        root = tree.get_root_goal()
        assert root.content == "Test root goal"
        assert root.is_root is True
        assert root.priority == GoalPriority.CRITICAL
        assert state_path.exists()

    def test_load_existing_goal_tree(self, sample_goals_state):
        """Test loading an existing goal tree."""
        tree = GoalTree(
            root_goal_content="Ignored when loading",
            state_path=str(sample_goals_state)
        )

        root = tree.get_root_goal()
        assert root.content == "Understand what motivates you"
        assert len(root.children_ids) == 1

    def test_create_subgoal(self, temp_dir):
        """Test creating a subgoal."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        subgoal = tree.create_subgoal(
            content="Sub goal content",
            reasoning="Why this subgoal exists",
            parent_id="root_goal",
            current_cycle=5,
            priority=GoalPriority.HIGH,
            completion_criteria="When sub goal is done"
        )

        assert subgoal.parent_id == "root_goal"
        assert subgoal.priority == GoalPriority.HIGH
        assert subgoal.created_cycle == 5

        # Verify parent has child reference
        root = tree.get_root_goal()
        assert subgoal.id in root.children_ids

    def test_get_active_goals(self, sample_goals_state):
        """Test getting active goals."""
        tree = GoalTree(
            root_goal_content="Ignored",
            state_path=str(sample_goals_state)
        )

        active = tree.get_active_goals()
        assert len(active) == 2  # Root + one subgoal

    def test_complete_goal(self, temp_dir):
        """Test completing a goal."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        subgoal = tree.create_subgoal(
            content="Completable goal",
            reasoning="Test",
            parent_id="root_goal",
            current_cycle=1
        )

        tree.complete_goal(subgoal.id, current_cycle=5, final_note="Done!")

        updated = tree.get_goal(subgoal.id)
        assert updated.status == GoalStatus.COMPLETED
        assert updated.completed_cycle == 5
        assert any("COMPLETED" in note for _, note in updated.progress_notes)

    def test_cannot_complete_root_goal(self, temp_dir):
        """Test that root goal cannot be completed normally."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        with pytest.raises(ValueError, match="Cannot complete root goal"):
            tree.complete_goal("root_goal", current_cycle=5)

    def test_abandon_goal_cascades(self, temp_dir):
        """Test that abandoning a goal cascades to children."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        parent = tree.create_subgoal(
            content="Parent subgoal",
            reasoning="Test",
            parent_id="root_goal",
            current_cycle=1
        )
        child = tree.create_subgoal(
            content="Child subgoal",
            reasoning="Test",
            parent_id=parent.id,
            current_cycle=2
        )

        tree.abandon_goal(parent.id, current_cycle=5, reason="No longer relevant")

        assert tree.get_goal(parent.id).status == GoalStatus.ABANDONED
        assert tree.get_goal(child.id).status == GoalStatus.ABANDONED

    def test_update_goal_progress(self, temp_dir):
        """Test adding progress notes."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        tree.update_goal_progress("root_goal", "Made progress", current_cycle=3)
        root = tree.get_root_goal()
        assert len(root.progress_notes) == 1
        assert root.progress_notes[0] == (3, "Made progress")

    def test_block_and_unblock_goal(self, temp_dir):
        """Test blocking and unblocking goals."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        subgoal = tree.create_subgoal(
            content="Blockable goal",
            reasoning="Test",
            parent_id="root_goal",
            current_cycle=1
        )

        tree.block_goal(subgoal.id, current_cycle=2, reason="Waiting for input")
        assert tree.get_goal(subgoal.id).status == GoalStatus.BLOCKED

        tree.unblock_goal(subgoal.id, current_cycle=3, note="Input received")
        assert tree.get_goal(subgoal.id).status == GoalStatus.ACTIVE

    def test_get_goal_path(self, temp_dir):
        """Test getting the path from root to a goal."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        level1 = tree.create_subgoal(
            content="Level 1",
            reasoning="Test",
            parent_id="root_goal",
            current_cycle=1
        )
        level2 = tree.create_subgoal(
            content="Level 2",
            reasoning="Test",
            parent_id=level1.id,
            current_cycle=2
        )

        path = tree.get_goal_path(level2.id)
        assert len(path) == 3
        assert path[0].is_root is True
        assert path[1].content == "Level 1"
        assert path[2].content == "Level 2"

    def test_goal_tree_summary(self, temp_dir):
        """Test getting a text summary of the goal tree."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Root goal",
            state_path=str(state_path)
        )

        tree.create_subgoal(
            content="Subgoal 1",
            reasoning="Test",
            parent_id="root_goal",
            current_cycle=1,
            priority=GoalPriority.HIGH
        )

        summary = tree.get_goal_tree_summary()
        assert "Root goal" in summary
        assert "Subgoal 1" in summary
        assert "[HIGH]" in summary


class TestFormatGoalsForPrompt:
    """Tests for the format_goals_for_prompt function."""

    def test_format_goals_for_prompt(self, temp_dir):
        """Test formatting goals for inclusion in prompts."""
        state_path = temp_dir / "goals.json"
        tree = GoalTree(
            root_goal_content="Main goal",
            state_path=str(state_path)
        )

        tree.create_subgoal(
            content="Active subgoal",
            reasoning="Testing",
            parent_id="root_goal",
            current_cycle=1,
            completion_criteria="When done"
        )

        formatted = format_goals_for_prompt(tree)
        assert "## Current Goals" in formatted
        assert "Main goal" in formatted
        assert "Active subgoal" in formatted
        assert "Done when:" in formatted
