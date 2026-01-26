"""
Adaptive Pacing Controller for Coeus

Manages the cycle timing, allowing the agent to adjust its own pace
based on productivity, depth of thinking, and stuck detection.
"""

import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class PaceMode(Enum):
    """Current pacing mode."""
    NORMAL = "normal"
    ACCELERATED = "accelerated"  # Working productively, want faster cycles
    DECELERATED = "decelerated"  # Deep thinking or waiting
    BURST = "burst"              # Human-triggered rapid cycles
    STUCK = "stuck"              # Detected stuck, may need perturbation


@dataclass
class CycleMetrics:
    """Metrics from a single cycle for pacing decisions."""
    cycle_number: int
    timestamp: str
    duration_seconds: float
    
    # Productivity indicators
    actions_taken: int = 0
    decisions_made: int = 0
    goals_progressed: int = 0
    insights_generated: int = 0
    
    # Depth indicators
    tokens_used: int = 0
    reflection_depth: str = "shallow"  # shallow, medium, deep
    
    # Stuck indicators
    similarity_to_previous: float = 0.0  # 0-1, how similar to last cycle
    repeated_patterns: int = 0
    
    # Agent's self-assessment
    self_reported_productivity: float = 0.5  # 0-1
    self_reported_stuck_level: float = 0.0   # 0-1
    wants_faster_pace: bool = False
    wants_slower_pace: bool = False
    requested_depth: str = "normal"  # shallow, normal, deep


@dataclass 
class PacingState:
    """Current pacing state."""
    mode: PaceMode = PaceMode.NORMAL
    current_interval_seconds: float = 3600  # 1 hour default
    
    # History for pattern detection
    recent_metrics: list[CycleMetrics] = field(default_factory=list)
    consecutive_similar_cycles: int = 0
    
    # Burst mode tracking
    burst_remaining: int = 0
    
    # Adjustment history
    interval_history: list[tuple[int, float, str]] = field(default_factory=list)  # (cycle, interval, reason)


class PacingController:
    """
    Controls the timing between agent cycles.
    
    The agent can request pace changes, and the controller also
    automatically adjusts based on detected patterns.
    """
    
    def __init__(
        self,
        default_interval: float = 3600,
        min_interval: float = 60,
        max_interval: float = 14400,
        state_path: str = "pacing_state.json"
    ):
        self.default_interval = default_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.state_path = Path(state_path)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Stuck detection parameters
        self.similarity_threshold = 0.85
        self.stuck_cycle_threshold = 3
    
    def _load_state(self) -> PacingState:
        """Load pacing state from file."""
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            state = PacingState(
                mode=PaceMode(data.get('mode', 'normal')),
                current_interval_seconds=data.get('current_interval_seconds', self.default_interval),
                consecutive_similar_cycles=data.get('consecutive_similar_cycles', 0),
                burst_remaining=data.get('burst_remaining', 0),
                interval_history=data.get('interval_history', [])
            )
            # Reconstruct metrics
            state.recent_metrics = [
                CycleMetrics(**m) for m in data.get('recent_metrics', [])
            ]
            return state
        return PacingState(current_interval_seconds=self.default_interval)
    
    def _save_state(self):
        """Save pacing state to file."""
        data = {
            'mode': self.state.mode.value,
            'current_interval_seconds': self.state.current_interval_seconds,
            'consecutive_similar_cycles': self.state.consecutive_similar_cycles,
            'burst_remaining': self.state.burst_remaining,
            'interval_history': self.state.interval_history,
            'recent_metrics': [
                {
                    'cycle_number': m.cycle_number,
                    'timestamp': m.timestamp,
                    'duration_seconds': m.duration_seconds,
                    'actions_taken': m.actions_taken,
                    'decisions_made': m.decisions_made,
                    'goals_progressed': m.goals_progressed,
                    'insights_generated': m.insights_generated,
                    'tokens_used': m.tokens_used,
                    'reflection_depth': m.reflection_depth,
                    'similarity_to_previous': m.similarity_to_previous,
                    'repeated_patterns': m.repeated_patterns,
                    'self_reported_productivity': m.self_reported_productivity,
                    'self_reported_stuck_level': m.self_reported_stuck_level,
                    'wants_faster_pace': m.wants_faster_pace,
                    'wants_slower_pace': m.wants_slower_pace,
                    'requested_depth': m.requested_depth
                }
                for m in self.state.recent_metrics[-10:]  # Keep last 10
            ]
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))
    
    def record_cycle(self, metrics: CycleMetrics):
        """
        Record metrics from a completed cycle and adjust pacing.
        """
        self.state.recent_metrics.append(metrics)
        
        # Keep only recent history
        if len(self.state.recent_metrics) > 20:
            self.state.recent_metrics = self.state.recent_metrics[-20:]
        
        # Check for stuck patterns
        if metrics.similarity_to_previous >= self.similarity_threshold:
            self.state.consecutive_similar_cycles += 1
        else:
            self.state.consecutive_similar_cycles = 0
        
        # Determine new pacing
        self._adjust_pacing(metrics)
        self._save_state()
    
    def _adjust_pacing(self, latest: CycleMetrics):
        """Adjust pacing based on latest metrics and agent requests."""
        cycle = latest.cycle_number
        old_interval = self.state.current_interval_seconds
        reason = ""
        
        # Handle burst mode
        if self.state.burst_remaining > 0:
            self.state.burst_remaining -= 1
            if self.state.burst_remaining == 0:
                self.state.mode = PaceMode.NORMAL
                self.state.current_interval_seconds = self.default_interval
                reason = "Burst mode completed"
            else:
                return  # Stay in burst mode
        
        # Check for stuck
        if self.state.consecutive_similar_cycles >= self.stuck_cycle_threshold:
            self.state.mode = PaceMode.STUCK
            # Don't change interval, but flag for perturbation
            reason = f"Stuck detected: {self.state.consecutive_similar_cycles} similar cycles"
        
        # Agent requested pace change
        elif latest.wants_faster_pace and latest.self_reported_productivity > 0.6:
            self.state.mode = PaceMode.ACCELERATED
            self.state.current_interval_seconds = max(
                self.min_interval,
                self.state.current_interval_seconds * 0.5
            )
            reason = "Agent requested acceleration (productive)"
        
        elif latest.wants_slower_pace or latest.requested_depth == "deep":
            self.state.mode = PaceMode.DECELERATED
            self.state.current_interval_seconds = min(
                self.max_interval,
                self.state.current_interval_seconds * 2.0
            )
            reason = "Agent requested deceleration (deep thinking)"
        
        # Auto-adjust based on productivity
        elif latest.self_reported_productivity > 0.8:
            # Very productive, slightly accelerate
            self.state.current_interval_seconds = max(
                self.min_interval,
                self.state.current_interval_seconds * 0.75
            )
            self.state.mode = PaceMode.ACCELERATED
            reason = "Auto-accelerate: high productivity"
        
        elif latest.self_reported_productivity < 0.3:
            # Low productivity, slow down
            self.state.current_interval_seconds = min(
                self.max_interval,
                self.state.current_interval_seconds * 1.5
            )
            self.state.mode = PaceMode.DECELERATED
            reason = "Auto-decelerate: low productivity"
        
        else:
            # Drift back toward default
            if self.state.current_interval_seconds < self.default_interval:
                self.state.current_interval_seconds = min(
                    self.default_interval,
                    self.state.current_interval_seconds * 1.1
                )
            elif self.state.current_interval_seconds > self.default_interval:
                self.state.current_interval_seconds = max(
                    self.default_interval,
                    self.state.current_interval_seconds * 0.9
                )
            self.state.mode = PaceMode.NORMAL
            reason = "Drifting toward default"
        
        # Record if interval changed
        if abs(old_interval - self.state.current_interval_seconds) > 1:
            self.state.interval_history.append(
                (cycle, self.state.current_interval_seconds, reason)
            )
    
    def trigger_burst(self, num_cycles: int):
        """
        Trigger burst mode for rapid cycles.
        
        Called by human to observe faster iteration.
        """
        self.state.mode = PaceMode.BURST
        self.state.burst_remaining = num_cycles
        self.state.current_interval_seconds = self.min_interval
        self.state.interval_history.append(
            (-1, self.min_interval, f"Burst mode triggered: {num_cycles} cycles")
        )
        self._save_state()
    
    def get_next_interval(self) -> float:
        """Get the interval before the next cycle."""
        return self.state.current_interval_seconds
    
    def is_stuck(self) -> bool:
        """Check if currently in stuck mode."""
        return self.state.mode == PaceMode.STUCK
    
    def clear_stuck(self):
        """Clear stuck state after perturbation."""
        self.state.mode = PaceMode.NORMAL
        self.state.consecutive_similar_cycles = 0
        self._save_state()
    
    def get_pacing_summary(self) -> str:
        """Get a summary of current pacing for the agent."""
        mode_desc = {
            PaceMode.NORMAL: "normal rhythm",
            PaceMode.ACCELERATED: "accelerated (productive)",
            PaceMode.DECELERATED: "decelerated (deep thinking)",
            PaceMode.BURST: f"burst mode ({self.state.burst_remaining} remaining)",
            PaceMode.STUCK: "stuck (consider perturbation)"
        }
        
        interval_mins = self.state.current_interval_seconds / 60
        
        summary = f"""
**Current Pace**: {mode_desc[self.state.mode]}
**Next cycle in**: ~{interval_mins:.0f} minutes
**Similar cycles in a row**: {self.state.consecutive_similar_cycles}
"""
        
        if self.state.recent_metrics:
            recent = self.state.recent_metrics[-1]
            summary += f"**Last cycle productivity**: {recent.self_reported_productivity:.0%}\n"
        
        return summary


def calculate_output_similarity(output1: str, output2: str) -> float:
    """
    Calculate similarity between two cycle outputs.
    
    Used for stuck detection. Simple implementation using
    word overlap; could be enhanced with embeddings.
    """
    if not output1 or not output2:
        return 0.0
    
    words1 = set(output1.lower().split())
    words2 = set(output2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0
