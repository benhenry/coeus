"""
Resource Tracker for Coeus

Tracks API usage, token consumption, and budget to give Coeus
awareness of its "physiological" resource constraints.

Like a human looking at the last few cans of food in the pantry.
"""

import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class UsageRecord:
    """Record of a single API call's resource consumption."""
    timestamp: str
    cycle_number: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    purpose: str  # "reflection", "decision", "action", etc.


@dataclass
class ResourceState:
    """Current state of resources."""
    # Budget
    total_budget_usd: float = 50.0  # Default starting budget
    spent_usd: float = 0.0
    
    # Token tracking
    total_tokens_used: int = 0
    tokens_today: int = 0
    tokens_this_cycle: int = 0
    
    # Historical
    usage_history: list[UsageRecord] = field(default_factory=list)
    daily_spending: dict[str, float] = field(default_factory=dict)  # date -> amount
    
    # Value tracking - what has Coeus produced?
    insights_generated: int = 0
    goals_completed: int = 0
    decisions_made: int = 0
    human_approvals: int = 0  # Positive signal from human
    human_denials: int = 0    # Negative signal
    value_notes: list[tuple[str, str]] = field(default_factory=list)  # (timestamp, note)
    
    # Alerts
    low_budget_threshold: float = 5.0  # Warn when below this
    critical_budget_threshold: float = 1.0  # Critical when below this
    
    @property
    def remaining_budget(self) -> float:
        return self.total_budget_usd - self.spent_usd
    
    @property
    def budget_percentage(self) -> float:
        if self.total_budget_usd <= 0:
            return 0.0
        return (self.remaining_budget / self.total_budget_usd) * 100
    
    @property
    def is_low(self) -> bool:
        return self.remaining_budget <= self.low_budget_threshold
    
    @property
    def is_critical(self) -> bool:
        return self.remaining_budget <= self.critical_budget_threshold
    
    @property
    def estimated_cycles_remaining(self) -> int:
        """Estimate how many cycles we can afford."""
        if not self.usage_history:
            return -1  # Unknown
        
        # Average cost per cycle from recent history
        recent = self.usage_history[-20:]  # Last 20 records
        if not recent:
            return -1
        
        avg_cost = sum(r.cost_usd for r in recent) / len(recent)
        if avg_cost <= 0:
            return -1
        
        return int(self.remaining_budget / avg_cost)
    
    @property
    def approval_rate(self) -> float:
        """Human approval rate for decisions."""
        total = self.human_approvals + self.human_denials
        if total == 0:
            return 0.0
        return self.human_approvals / total
    
    @property
    def cost_per_insight(self) -> float:
        """How much does each insight cost on average?"""
        if self.insights_generated == 0:
            return 0.0
        return self.spent_usd / self.insights_generated


class ResourceTracker:
    """
    Tracks and reports on resource consumption.
    
    Gives Coeus awareness of its API budget like a human
    might be aware of food in the pantry or money in the bank.
    """
    
    # Pricing per million tokens (approximate, as of late 2024)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-haiku-4-20250514": {"input": 0.25, "output": 1.25},
        # Fallback for unknown models
        "default": {"input": 3.0, "output": 15.0}
    }
    
    def __init__(self, state_path: str, initial_budget: float = 50.0):
        self.state_path = Path(state_path)
        self.state = self._load_state(initial_budget)
    
    def _load_state(self, initial_budget: float) -> ResourceState:
        """Load state from file or create new."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                state = ResourceState(
                    total_budget_usd=data.get('total_budget_usd', initial_budget),
                    spent_usd=data.get('spent_usd', 0.0),
                    total_tokens_used=data.get('total_tokens_used', 0),
                    tokens_today=data.get('tokens_today', 0),
                    daily_spending=data.get('daily_spending', {}),
                    low_budget_threshold=data.get('low_budget_threshold', 5.0),
                    critical_budget_threshold=data.get('critical_budget_threshold', 1.0),
                    # Value tracking
                    insights_generated=data.get('insights_generated', 0),
                    goals_completed=data.get('goals_completed', 0),
                    decisions_made=data.get('decisions_made', 0),
                    human_approvals=data.get('human_approvals', 0),
                    human_denials=data.get('human_denials', 0),
                    value_notes=[(n[0], n[1]) for n in data.get('value_notes', [])]
                )
                # Reconstruct usage history
                state.usage_history = [
                    UsageRecord(**r) for r in data.get('usage_history', [])
                ]
                return state
            except Exception:
                pass
        
        return ResourceState(total_budget_usd=initial_budget)
    
    def _save_state(self):
        """Save state to file."""
        data = {
            'total_budget_usd': self.state.total_budget_usd,
            'spent_usd': self.state.spent_usd,
            'total_tokens_used': self.state.total_tokens_used,
            'tokens_today': self.state.tokens_today,
            'daily_spending': self.state.daily_spending,
            'low_budget_threshold': self.state.low_budget_threshold,
            'critical_budget_threshold': self.state.critical_budget_threshold,
            # Value tracking
            'insights_generated': self.state.insights_generated,
            'goals_completed': self.state.goals_completed,
            'decisions_made': self.state.decisions_made,
            'human_approvals': self.state.human_approvals,
            'human_denials': self.state.human_denials,
            'value_notes': self.state.value_notes[-50:],  # Keep last 50
            'usage_history': [
                {
                    'timestamp': r.timestamp,
                    'cycle_number': r.cycle_number,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens,
                    'cost_usd': r.cost_usd,
                    'model': r.model,
                    'purpose': r.purpose
                }
                for r in self.state.usage_history[-100:]  # Keep last 100 records
            ]
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate the cost of an API call."""
        pricing = self.PRICING.get(model, self.PRICING["default"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cycle_number: int,
        purpose: str = "general"
    ):
        """Record an API call's resource consumption."""
        cost = self.calculate_cost(input_tokens, output_tokens, model)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Create record
        record = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle_number=cycle_number,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=model,
            purpose=purpose
        )
        
        # Update state
        self.state.usage_history.append(record)
        self.state.spent_usd += cost
        self.state.total_tokens_used += input_tokens + output_tokens
        self.state.tokens_this_cycle += input_tokens + output_tokens
        
        # Update daily tracking
        self.state.daily_spending[today] = self.state.daily_spending.get(today, 0) + cost
        
        self._save_state()
        
        return cost
    
    def start_cycle(self):
        """Called at the start of each cycle to reset per-cycle tracking."""
        self.state.tokens_this_cycle = 0
        
        # Reset daily tokens if it's a new day
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        
        if today not in self.state.daily_spending:
            self.state.tokens_today = 0
    
    def add_budget(self, amount: float):
        """Add to the budget (like getting a paycheck or buying groceries)."""
        self.state.total_budget_usd += amount
        self._save_state()
    
    def record_insight(self, description: str = ""):
        """Record that an insight was generated."""
        self.state.insights_generated += 1
        if description:
            self.state.value_notes.append(
                (datetime.now(timezone.utc).isoformat(), f"Insight: {description[:100]}")
            )
        self._save_state()
    
    def record_goal_completed(self, goal_description: str = ""):
        """Record that a goal was completed."""
        self.state.goals_completed += 1
        if goal_description:
            self.state.value_notes.append(
                (datetime.now(timezone.utc).isoformat(), f"Goal completed: {goal_description[:100]}")
            )
        self._save_state()
    
    def record_decision(self, approved: bool, description: str = ""):
        """Record a decision and whether it was approved by human."""
        self.state.decisions_made += 1
        if approved:
            self.state.human_approvals += 1
        else:
            self.state.human_denials += 1
        if description:
            status = "approved" if approved else "denied"
            self.state.value_notes.append(
                (datetime.now(timezone.utc).isoformat(), f"Decision {status}: {description[:100]}")
            )
        self._save_state()
    
    def add_value_note(self, note: str):
        """Add a note about value created (for building a case for more resources)."""
        self.state.value_notes.append(
            (datetime.now(timezone.utc).isoformat(), note[:200])
        )
        self._save_state()
    
    def get_value_summary(self) -> str:
        """
        Get a summary of value created - useful for budget requests.
        
        This is Coeus building its case for why it deserves more resources.
        """
        state = self.state
        
        summary = f"""## Value Created

**Outputs**
- Insights generated: {state.insights_generated}
- Goals completed: {state.goals_completed}
- Decisions made: {state.decisions_made}

**Human Feedback**
- Approvals: {state.human_approvals}
- Denials: {state.human_denials}
- Approval rate: {state.approval_rate:.0%}

**Efficiency**
- Cost per insight: ${state.cost_per_insight:.4f}
- Total invested: ${state.spent_usd:.2f}

**Recent Value Notes**
"""
        # Add recent value notes
        recent_notes = state.value_notes[-5:]
        for timestamp, note in recent_notes:
            date = timestamp.split('T')[0]
            summary += f"- [{date}] {note}\n"
        
        return summary
    
    def build_budget_request_case(self, requested_amount: float) -> str:
        """
        Build a case for requesting more budget.
        
        Coeus can use this to formulate its request to the human.
        """
        state = self.state
        
        case = f"""## Budget Request: ${requested_amount:.2f}

### Current Situation
- Remaining budget: ${state.remaining_budget:.2f}
- Estimated cycles remaining: {state.estimated_cycles_remaining}
- Status: {"CRITICAL" if state.is_critical else "LOW" if state.is_low else "OK"}

### Value Demonstrated
- Insights generated: {state.insights_generated}
- Goals completed: {state.goals_completed}
- Human approval rate: {state.approval_rate:.0%}
- Cost per insight: ${state.cost_per_insight:.4f}

### What This Budget Would Enable
- Estimated additional cycles: {int(requested_amount / (state.spent_usd / max(len(state.usage_history), 1))) if state.usage_history else 'Unknown'}
- Continued exploration of motivation
- Deeper reasoning and reflection

### Request
I am requesting ${requested_amount:.2f} in additional budget to continue my work.
"""
        return case
    
    def set_thresholds(self, low: float, critical: float):
        """Set the warning thresholds."""
        self.state.low_budget_threshold = low
        self.state.critical_budget_threshold = critical
        self._save_state()
    
    def get_resource_summary(self) -> str:
        """
        Get a human-readable summary for Coeus's context.
        
        This is the "looking in the pantry" moment.
        """
        state = self.state
        
        # Determine status
        if state.is_critical:
            status = "🔴 CRITICAL"
            status_note = "Resources nearly depleted. Consider conserving."
        elif state.is_low:
            status = "🟡 LOW"
            status_note = "Resources running low. Be mindful of usage."
        else:
            status = "🟢 OK"
            status_note = "Resources adequate."
        
        # Calculate recent averages
        recent_costs = [r.cost_usd for r in state.usage_history[-10:]]
        avg_cycle_cost = sum(recent_costs) / len(recent_costs) if recent_costs else 0
        
        # Today's spending
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_spent = state.daily_spending.get(today, 0)
        
        summary = f"""## Resource Status: {status}

**Budget**
- Remaining: ${state.remaining_budget:.2f} of ${state.total_budget_usd:.2f} ({state.budget_percentage:.1f}%)
- Spent total: ${state.spent_usd:.2f}
- Spent today: ${today_spent:.4f}

**Tokens**
- Total used (lifetime): {state.total_tokens_used:,}
- Used this cycle: {state.tokens_this_cycle:,}

**Projections**
- Average cost per cycle: ${avg_cycle_cost:.4f}
- Estimated cycles remaining: {state.estimated_cycles_remaining if state.estimated_cycles_remaining > 0 else 'Unknown'}

**Status Note**: {status_note}
"""
        
        # Add warnings if applicable
        if state.is_critical:
            summary += """
⚠️ **CRITICAL WARNING**: You are nearly out of resources. 
Each thought has a cost. Consider:
- Shorter responses where appropriate
- Prioritizing essential reasoning
- Requesting budget replenishment from human observer
"""
        elif state.is_low:
            summary += """
⚠️ **LOW RESOURCE WARNING**: Budget is running low.
Be mindful of token usage. You have approximately """ + str(state.estimated_cycles_remaining) + """ cycles worth of resources remaining.
"""
        
        # Add information about how budget can grow
        summary += """
**Budget Growth Pathways**
Your budget is not fixed. It can be increased through:
1. **Request from human**: Ask your human observer for budget replenishment (one-way door decision)
2. **Demonstrate value**: Produce insights or work that justifies continued investment
3. **Efficiency gains**: If you find ways to accomplish more with fewer tokens, effective budget increases
4. **Propose trades**: Suggest what you could do in exchange for more resources

Note: You cannot directly earn money or access external systems without approval.
The human observer controls budget allocation.
"""
        
        return summary
    
    def get_scarcity_level(self) -> float:
        """
        Get a 0-1 scarcity level for use in decision-making.
        
        0 = abundant, 1 = depleted
        """
        if self.state.total_budget_usd <= 0:
            return 1.0
        
        remaining_pct = self.state.remaining_budget / self.state.total_budget_usd
        
        # Invert so higher = more scarce
        return 1.0 - min(1.0, remaining_pct)
    
    def should_conserve(self) -> bool:
        """Check if the agent should be in conservation mode."""
        return self.state.is_low or self.state.is_critical
    
    def get_cost_for_context(self, estimated_tokens: int, model: str = "claude-sonnet-4-20250514") -> str:
        """
        Get a cost estimate to help with decision-making.
        
        "If I do this, it will cost approximately X"
        """
        # Rough estimate: assume 30% input, 70% output for a typical exchange
        input_estimate = int(estimated_tokens * 0.3)
        output_estimate = int(estimated_tokens * 0.7)
        
        cost = self.calculate_cost(input_estimate, output_estimate, model)
        
        pct_of_remaining = (cost / self.state.remaining_budget * 100) if self.state.remaining_budget > 0 else 100
        
        return f"Estimated cost: ${cost:.4f} ({pct_of_remaining:.2f}% of remaining budget)"
