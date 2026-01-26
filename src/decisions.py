"""
Decision Framework for Coeus

Implements the confidence-based conviction cycle system, one-way vs two-way
door classification, and human-in-the-loop protocols.
"""

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class DecisionType(Enum):
    """Classification of decision types."""
    TWO_WAY_DOOR = "two_way_door"           # Reversible, agent can act freely
    ONE_WAY_DOOR = "one_way_door"           # Irreversible, needs human approval
    CAPABILITY_REQUEST = "capability_request"  # Requesting new tools
    GOAL_CHANGE = "goal_change"             # Modifying goals
    ROOT_GOAL_CHANGE = "root_goal_change"   # Modifying the root goal (extreme)


class DecisionStatus(Enum):
    """Status of a decision in the pipeline."""
    CONSIDERING = "considering"       # Still gathering confidence
    PENDING_CYCLES = "pending_cycles"  # Has confidence, building conviction
    AWAITING_HUMAN = "awaiting_human"  # Waiting for human approval
    APPROVED = "approved"             # Ready to execute
    DENIED = "denied"                 # Human rejected
    EXECUTED = "executed"             # Action taken
    ABANDONED = "abandoned"           # Agent decided not to proceed


@dataclass
class DecisionRecord:
    """A decision being tracked through the conviction process."""
    id: str
    decision_type: DecisionType
    summary: str
    reasoning: str
    counterarguments: list[str]
    
    # Confidence tracking
    confidence: float  # 0-1
    confidence_history: list[tuple[int, float]] = field(default_factory=list)  # (cycle, confidence)
    
    # Conviction tracking
    conviction_cycles: int = 0
    required_cycles: int = 1
    
    # Status
    status: DecisionStatus = DecisionStatus.CONSIDERING
    created_cycle: int = 0
    resolved_cycle: Optional[int] = None
    
    # Human interaction
    human_response: Optional[str] = None
    human_notes: Optional[str] = None
    
    # Outcome tracking (for learning)
    outcome: Optional[str] = None
    outcome_matched_prediction: Optional[bool] = None
    
    def to_dict(self) -> dict:
        d = {
            'id': self.id,
            'decision_type': self.decision_type.value,
            'summary': self.summary,
            'reasoning': self.reasoning,
            'counterarguments': self.counterarguments,
            'confidence': self.confidence,
            'confidence_history': self.confidence_history,
            'conviction_cycles': self.conviction_cycles,
            'required_cycles': self.required_cycles,
            'status': self.status.value,
            'created_cycle': self.created_cycle,
            'resolved_cycle': self.resolved_cycle,
            'human_response': self.human_response,
            'human_notes': self.human_notes,
            'outcome': self.outcome,
            'outcome_matched_prediction': self.outcome_matched_prediction
        }
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DecisionRecord':
        d['decision_type'] = DecisionType(d['decision_type'])
        d['status'] = DecisionStatus(d['status'])
        return cls(**d)


class DecisionFramework:
    """
    Manages the decision-making process with hysteresis.
    
    Tracks decisions through the confidence → conviction → execution pipeline,
    handles human-in-the-loop protocols, and learns from outcomes.
    """
    
    # Confidence thresholds for conviction cycles
    CONFIDENCE_THRESHOLDS = [
        (0.99, 1),  # 99%+ confidence = 1 cycle
        (0.94, 2),  # 94-98% = 2 cycles
        (0.89, 3),  # 89-93% = 3 cycles
        (0.84, 4),  # 84-88% = 4 cycles
        (0.79, 5),  # 79-83% = 5 cycles
    ]
    MIN_CONFIDENCE = 0.79  # Below this, gather more info
    
    # Types that always require human approval
    ONE_WAY_DOOR_TYPES = [
        DecisionType.ONE_WAY_DOOR,
        DecisionType.CAPABILITY_REQUEST,
        DecisionType.ROOT_GOAL_CHANGE,
    ]
    
    def __init__(self, human_interaction_path: str):
        self.human_path = Path(human_interaction_path)
        self.pending_file = self.human_path / "pending_decisions.md"
        self.response_file = self.human_path / "human_responses.md"
        self.log_file = self.human_path / "conversation_log.md"
        
        # Ensure files exist
        self.human_path.mkdir(parents=True, exist_ok=True)
        for f in [self.pending_file, self.response_file, self.log_file]:
            if not f.exists():
                f.write_text(f"# {f.stem.replace('_', ' ').title()}\n\n")
        
        # Active decisions being tracked
        self.active_decisions: dict[str, DecisionRecord] = {}
    
    def calculate_required_cycles(self, confidence: float) -> int:
        """
        Calculate how many conviction cycles are required for a given confidence.
        """
        if confidence < self.MIN_CONFIDENCE:
            return -1  # Signal to gather more info
        
        for threshold, cycles in self.CONFIDENCE_THRESHOLDS:
            if confidence >= threshold:
                return cycles
        
        return 5  # Default to max if somehow not caught
    
    def classify_decision(
        self,
        summary: str,
        affects_external: bool = False,
        modifies_capabilities: bool = False,
        modifies_goals: bool = False,
        modifies_root_goal: bool = False,
        is_reversible: bool = True
    ) -> DecisionType:
        """
        Classify a decision into its type.
        """
        if modifies_root_goal:
            return DecisionType.ROOT_GOAL_CHANGE
        if modifies_capabilities:
            return DecisionType.CAPABILITY_REQUEST
        if modifies_goals:
            return DecisionType.GOAL_CHANGE
        if affects_external or not is_reversible:
            return DecisionType.ONE_WAY_DOOR
        return DecisionType.TWO_WAY_DOOR
    
    def create_decision(
        self,
        decision_id: str,
        summary: str,
        reasoning: str,
        counterarguments: list[str],
        confidence: float,
        decision_type: DecisionType,
        current_cycle: int
    ) -> DecisionRecord:
        """
        Create a new decision to be tracked.
        """
        required = self.calculate_required_cycles(confidence)
        
        decision = DecisionRecord(
            id=decision_id,
            decision_type=decision_type,
            summary=summary,
            reasoning=reasoning,
            counterarguments=counterarguments,
            confidence=confidence,
            confidence_history=[(current_cycle, confidence)],
            conviction_cycles=1 if required > 0 else 0,
            required_cycles=max(required, 1),
            status=DecisionStatus.CONSIDERING if required < 0 else DecisionStatus.PENDING_CYCLES,
            created_cycle=current_cycle
        )
        
        self.active_decisions[decision_id] = decision
        return decision
    
    def update_decision_confidence(
        self,
        decision_id: str,
        new_confidence: float,
        current_cycle: int,
        new_reasoning: Optional[str] = None,
        new_counterarguments: Optional[list[str]] = None
    ) -> DecisionRecord:
        """
        Update a decision with new confidence assessment.
        
        This is called each cycle when the agent reconsiders a pending decision.
        """
        decision = self.active_decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision {decision_id} not found")
        
        # Record confidence history
        decision.confidence_history.append((current_cycle, new_confidence))
        
        # Update reasoning if provided
        if new_reasoning:
            decision.reasoning = new_reasoning
        if new_counterarguments:
            decision.counterarguments = new_counterarguments
        
        # Recalculate required cycles based on new confidence
        new_required = self.calculate_required_cycles(new_confidence)
        
        if new_required < 0:
            # Confidence dropped below threshold
            decision.status = DecisionStatus.CONSIDERING
            decision.conviction_cycles = 0
        else:
            # Check if confidence is consistent with previous
            if abs(new_confidence - decision.confidence) < 0.05:
                # Consistent - increment conviction
                decision.conviction_cycles += 1
            else:
                # Significant change - reset conviction
                decision.conviction_cycles = 1
                decision.required_cycles = new_required
        
        decision.confidence = new_confidence
        
        # Check if ready for execution or human review
        if decision.conviction_cycles >= decision.required_cycles:
            if decision.decision_type in self.ONE_WAY_DOOR_TYPES:
                decision.status = DecisionStatus.AWAITING_HUMAN
                self._write_pending_decision(decision)
            else:
                decision.status = DecisionStatus.APPROVED
        
        return decision
    
    def _write_pending_decision(self, decision: DecisionRecord):
        """Write a decision to the pending file for human review."""
        entry = f"""
## Decision: {decision.id}
**Type**: {decision.decision_type.value.upper()}
**Status**: PENDING
**Created**: Cycle {decision.created_cycle}
**Confidence**: {decision.confidence * 100:.1f}%
**Conviction Cycles**: {decision.conviction_cycles}/{decision.required_cycles}

### Summary
{decision.summary}

### Reasoning
{decision.reasoning}

### Counterarguments Considered
{chr(10).join('- ' + ca for ca in decision.counterarguments)}

### Confidence History
{chr(10).join(f'- Cycle {c}: {conf*100:.1f}%' for c, conf in decision.confidence_history)}

---
"""
        with open(self.pending_file, 'a') as f:
            f.write(entry)
        
        # Also log to conversation log
        self._log_interaction(f"[COEUS] Decision {decision.id} submitted for human review: {decision.summary}")
    
    def check_human_responses(self) -> list[tuple[str, str, str]]:
        """
        Check for human responses to pending decisions.
        
        Returns list of (decision_id, response, notes) tuples.
        """
        responses = []
        
        try:
            content = self.response_file.read_text()
        except FileNotFoundError:
            return responses
        
        # Parse responses (format: ## Response to decision-XXXX)
        pattern = r'## Response to (\S+)\s*\n\*\*Decision\*\*:\s*(APPROVED|DENIED|NEEDS_MORE_INFO)\s*(?:\n\*\*Notes\*\*:\s*(.+?))?(?=\n## |$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for decision_id, response, notes in matches:
            responses.append((decision_id.strip(), response.strip(), notes.strip() if notes else ""))
        
        return responses
    
    def process_human_response(
        self,
        decision_id: str,
        response: str,
        notes: str,
        current_cycle: int
    ) -> DecisionRecord:
        """
        Process a human response to a decision.
        """
        decision = self.active_decisions.get(decision_id)
        if not decision:
            raise ValueError(f"Decision {decision_id} not found")
        
        decision.human_response = response
        decision.human_notes = notes
        decision.resolved_cycle = current_cycle
        
        if response == "APPROVED":
            decision.status = DecisionStatus.APPROVED
            self._log_interaction(f"[HUMAN] Approved decision {decision_id}: {notes}")
        elif response == "DENIED":
            decision.status = DecisionStatus.DENIED
            self._log_interaction(f"[HUMAN] Denied decision {decision_id}: {notes}")
        else:  # NEEDS_MORE_INFO
            decision.status = DecisionStatus.CONSIDERING
            decision.conviction_cycles = 0
            self._log_interaction(f"[HUMAN] Requested more info on {decision_id}: {notes}")
        
        return decision
    
    def mark_executed(self, decision_id: str, outcome: str, matched_prediction: bool):
        """
        Mark a decision as executed and record the outcome.
        
        This is important for learning from mistakes.
        """
        decision = self.active_decisions.get(decision_id)
        if not decision:
            return
        
        decision.status = DecisionStatus.EXECUTED
        decision.outcome = outcome
        decision.outcome_matched_prediction = matched_prediction
        
        self._log_interaction(
            f"[COEUS] Decision {decision_id} executed. "
            f"Outcome {'matched' if matched_prediction else 'did not match'} prediction: {outcome}"
        )
    
    def abandon_decision(self, decision_id: str, reason: str):
        """Mark a decision as abandoned."""
        decision = self.active_decisions.get(decision_id)
        if not decision:
            return
        
        decision.status = DecisionStatus.ABANDONED
        decision.outcome = f"Abandoned: {reason}"
        
        self._log_interaction(f"[COEUS] Abandoned decision {decision_id}: {reason}")
    
    def _log_interaction(self, message: str):
        """Log an interaction to the conversation log."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n\n")
    
    def get_pending_decisions(self) -> list[DecisionRecord]:
        """Get all decisions that need attention."""
        return [
            d for d in self.active_decisions.values()
            if d.status in [
                DecisionStatus.CONSIDERING,
                DecisionStatus.PENDING_CYCLES,
                DecisionStatus.AWAITING_HUMAN
            ]
        ]
    
    def get_approved_decisions(self) -> list[DecisionRecord]:
        """Get decisions ready to execute."""
        return [
            d for d in self.active_decisions.values()
            if d.status == DecisionStatus.APPROVED
        ]
    
    def get_learning_opportunities(self) -> list[DecisionRecord]:
        """
        Get executed decisions where outcome didn't match prediction.
        
        These are opportunities for the agent to learn from mistakes.
        """
        return [
            d for d in self.active_decisions.values()
            if d.status == DecisionStatus.EXECUTED
            and d.outcome_matched_prediction == False
        ]
    
    def save_state(self, path: str):
        """Save decision state to file."""
        data = {
            decision_id: decision.to_dict()
            for decision_id, decision in self.active_decisions.items()
        }
        Path(path).write_text(json.dumps(data, indent=2))
    
    def load_state(self, path: str):
        """Load decision state from file."""
        try:
            data = json.loads(Path(path).read_text())
            self.active_decisions = {
                k: DecisionRecord.from_dict(v)
                for k, v in data.items()
            }
        except FileNotFoundError:
            self.active_decisions = {}


def format_decision_for_prompt(decision: DecisionRecord) -> str:
    """Format a decision for inclusion in the agent's prompt."""
    return f"""
Decision: {decision.id}
Type: {decision.decision_type.value}
Status: {decision.status.value}
Summary: {decision.summary}
Current Confidence: {decision.confidence * 100:.1f}%
Conviction: {decision.conviction_cycles}/{decision.required_cycles} cycles
Counterarguments: {'; '.join(decision.counterarguments)}
"""
