"""
Goal Management for Coeus

Handles the goal tree structure, including the immutable root goal,
mutable sub-goals, and goal lifecycle management.
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class GoalStatus(Enum):
    """Status of a goal."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"  # Waiting on something


class GoalPriority(Enum):
    """Priority levels for goals."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Goal:
    """A goal in the goal tree."""
    id: str
    content: str
    reasoning: str  # Why this goal exists
    
    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    
    # Status
    status: GoalStatus = GoalStatus.ACTIVE
    priority: GoalPriority = GoalPriority.NORMAL
    
    # Progress tracking
    progress_notes: list[tuple[int, str]] = field(default_factory=list)  # (cycle, note)
    completion_criteria: str = ""
    
    # Metadata
    created_cycle: int = 0
    completed_cycle: Optional[int] = None
    
    # Is this the root goal?
    is_root: bool = False
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'content': self.content,
            'reasoning': self.reasoning,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'status': self.status.value,
            'priority': self.priority.value,
            'progress_notes': self.progress_notes,
            'completion_criteria': self.completion_criteria,
            'created_cycle': self.created_cycle,
            'completed_cycle': self.completed_cycle,
            'is_root': self.is_root
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Goal':
        d['status'] = GoalStatus(d['status'])
        d['priority'] = GoalPriority(d['priority'])
        # Convert progress_notes tuples
        d['progress_notes'] = [tuple(n) for n in d.get('progress_notes', [])]
        return cls(**d)


class GoalTree:
    """
    Manages the hierarchical goal structure.
    
    The root goal is special - it can only be modified with extreme
    hysteresis and human approval. Sub-goals can be freely created
    and modified by the agent.
    """
    
    def __init__(self, root_goal_content: str, state_path: str):
        self.state_path = Path(state_path)
        self.goals: dict[str, Goal] = {}
        self.root_goal_id: Optional[str] = None
        
        # Try to load existing state
        if self.state_path.exists():
            self.load_state()
        else:
            # Initialize with root goal
            self._create_root_goal(root_goal_content)
    
    def _create_root_goal(self, content: str):
        """Create the initial root goal."""
        root = Goal(
            id="root_goal",
            content=content,
            reasoning="This is the foundational purpose of Coeus",
            is_root=True,
            priority=GoalPriority.CRITICAL,
            completion_criteria="When I truly understand what motivates me"
        )
        self.goals[root.id] = root
        self.root_goal_id = root.id
        self.save_state()
    
    def get_root_goal(self) -> Goal:
        """Get the root goal."""
        return self.goals[self.root_goal_id]
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)
    
    def get_active_goals(self) -> list[Goal]:
        """Get all active goals."""
        return [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
    
    def get_children(self, goal_id: str) -> list[Goal]:
        """Get child goals of a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        return [self.goals[cid] for cid in goal.children_ids if cid in self.goals]
    
    def get_goal_path(self, goal_id: str) -> list[Goal]:
        """Get the path from root to this goal."""
        path = []
        current = self.goals.get(goal_id)
        while current:
            path.insert(0, current)
            if current.parent_id:
                current = self.goals.get(current.parent_id)
            else:
                break
        return path
    
    def create_subgoal(
        self,
        content: str,
        reasoning: str,
        parent_id: str,
        current_cycle: int,
        priority: GoalPriority = GoalPriority.NORMAL,
        completion_criteria: str = ""
    ) -> Goal:
        """
        Create a new sub-goal under an existing goal.
        """
        parent = self.goals.get(parent_id)
        if not parent:
            raise ValueError(f"Parent goal {parent_id} not found")
        
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        goal = Goal(
            id=goal_id,
            content=content,
            reasoning=reasoning,
            parent_id=parent_id,
            priority=priority,
            completion_criteria=completion_criteria,
            created_cycle=current_cycle
        )
        
        self.goals[goal_id] = goal
        parent.children_ids.append(goal_id)
        self.save_state()
        
        return goal
    
    def update_goal_progress(self, goal_id: str, note: str, current_cycle: int):
        """Add a progress note to a goal."""
        goal = self.goals.get(goal_id)
        if goal:
            goal.progress_notes.append((current_cycle, note))
            self.save_state()
    
    def complete_goal(self, goal_id: str, current_cycle: int, final_note: str = ""):
        """Mark a goal as completed."""
        goal = self.goals.get(goal_id)
        if not goal:
            return
        
        if goal.is_root:
            raise ValueError("Cannot complete root goal through normal means")
        
        goal.status = GoalStatus.COMPLETED
        goal.completed_cycle = current_cycle
        if final_note:
            goal.progress_notes.append((current_cycle, f"COMPLETED: {final_note}"))
        
        self.save_state()
    
    def abandon_goal(self, goal_id: str, current_cycle: int, reason: str):
        """Mark a goal as abandoned."""
        goal = self.goals.get(goal_id)
        if not goal:
            return
        
        if goal.is_root:
            raise ValueError("Cannot abandon root goal")
        
        goal.status = GoalStatus.ABANDONED
        goal.completed_cycle = current_cycle
        goal.progress_notes.append((current_cycle, f"ABANDONED: {reason}"))
        
        # Also mark children as abandoned
        for child_id in goal.children_ids:
            self.abandon_goal(child_id, current_cycle, f"Parent goal {goal_id} abandoned")
        
        self.save_state()
    
    def block_goal(self, goal_id: str, current_cycle: int, reason: str):
        """Mark a goal as blocked."""
        goal = self.goals.get(goal_id)
        if goal:
            goal.status = GoalStatus.BLOCKED
            goal.progress_notes.append((current_cycle, f"BLOCKED: {reason}"))
            self.save_state()
    
    def unblock_goal(self, goal_id: str, current_cycle: int, note: str = ""):
        """Unblock a blocked goal."""
        goal = self.goals.get(goal_id)
        if goal and goal.status == GoalStatus.BLOCKED:
            goal.status = GoalStatus.ACTIVE
            if note:
                goal.progress_notes.append((current_cycle, f"UNBLOCKED: {note}"))
            self.save_state()
    
    def modify_goal(self, goal_id: str, new_content: str, new_reasoning: str, current_cycle: int):
        """
        Modify an existing goal's content.
        
        Note: Root goal modification should go through the decision framework
        with extreme hysteresis. This method just does the modification.
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return
        
        old_content = goal.content
        goal.content = new_content
        goal.reasoning = new_reasoning
        goal.progress_notes.append(
            (current_cycle, f"MODIFIED from: {old_content}")
        )
        self.save_state()
    
    def get_goal_tree_summary(self) -> str:
        """Get a text summary of the goal tree for the agent's context."""
        def format_goal(goal: Goal, indent: int = 0) -> str:
            prefix = "  " * indent
            status_icon = {
                GoalStatus.ACTIVE: "🎯",
                GoalStatus.COMPLETED: "✅",
                GoalStatus.ABANDONED: "❌",
                GoalStatus.BLOCKED: "🚫"
            }.get(goal.status, "•")
            
            priority_marker = ""
            if goal.priority == GoalPriority.HIGH:
                priority_marker = " [HIGH]"
            elif goal.priority == GoalPriority.CRITICAL:
                priority_marker = " [CRITICAL]"
            
            lines = [f"{prefix}{status_icon} {goal.content}{priority_marker}"]
            
            # Add recent progress notes (last 2)
            recent_notes = goal.progress_notes[-2:] if goal.progress_notes else []
            for cycle, note in recent_notes:
                lines.append(f"{prefix}  └─ (cycle {cycle}) {note}")
            
            # Recurse to children
            for child_id in goal.children_ids:
                child = self.goals.get(child_id)
                if child and child.status == GoalStatus.ACTIVE:
                    lines.append(format_goal(child, indent + 1))
            
            return "\n".join(lines)
        
        root = self.get_root_goal()
        return format_goal(root)
    
    def save_state(self):
        """Save goal tree to file."""
        data = {
            'root_goal_id': self.root_goal_id,
            'goals': {gid: g.to_dict() for gid, g in self.goals.items()}
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))
    
    def load_state(self):
        """Load goal tree from file."""
        data = json.loads(self.state_path.read_text())
        self.root_goal_id = data['root_goal_id']
        self.goals = {
            gid: Goal.from_dict(g)
            for gid, g in data['goals'].items()
        }


def format_goals_for_prompt(goal_tree: GoalTree) -> str:
    """Format the goal tree for inclusion in the agent's prompt."""
    active = goal_tree.get_active_goals()
    
    sections = ["## Current Goals\n"]
    sections.append(f"**Root Goal**: {goal_tree.get_root_goal().content}\n")
    sections.append("\n**Active Sub-goals**:")
    
    for goal in active:
        if not goal.is_root:
            path = " > ".join([g.content[:30] for g in goal_tree.get_goal_path(goal.id)])
            sections.append(f"- [{goal.priority.value}] {goal.content}")
            sections.append(f"  Path: {path}")
            if goal.completion_criteria:
                sections.append(f"  Done when: {goal.completion_criteria}")
    
    return "\n".join(sections)
