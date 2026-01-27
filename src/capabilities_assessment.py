"""
Capabilities Assessment for Coeus

Periodic self-assessment against predefined and agent-defined benchmarks.
Full assessment every N cycles (configurable), lightweight summary always
available in context.

Follows the ResourceTracker pattern.
"""

import json
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Benchmark:
    """A single capability benchmark."""
    id: str
    name: str
    description: str
    category: str
    source: str  # "predefined" or "agent_defined"
    target_level: float  # 0-1
    current_level: float = 0.0
    assessment_history: list[tuple[int, float, str]] = field(default_factory=list)
    # Each tuple: (cycle_number, level, notes)


@dataclass
class AssessmentResult:
    """Result of a full capabilities assessment."""
    cycle_number: int
    timestamp: str
    benchmarks_assessed: list[dict] = field(default_factory=list)
    overall_score: float = 0.0
    new_capabilities_desired: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    trend: str = "unknown"  # "improving", "stable", "declining", "unknown"


@dataclass
class AssessmentState:
    """Persistent state for capabilities assessment."""
    benchmarks: dict[str, Benchmark] = field(default_factory=dict)
    assessment_history: list[AssessmentResult] = field(default_factory=list)
    last_full_assessment_cycle: int = 0
    assessment_interval: int = 10


class CapabilitiesAssessor:
    """
    Tracks and assesses capabilities against benchmarks.

    Gives Coeus awareness of its own capability levels and trends,
    like a student tracking their progress across subjects.
    """

    def __init__(self, config: dict, state_path: str):
        self.state_path = Path(state_path)
        self.config = config
        self.state = self._load_state()

    def _load_state(self) -> AssessmentState:
        """Load state from file, merging with config benchmarks."""
        state = AssessmentState(
            assessment_interval=self.config.get('full_assessment_interval', 10)
        )

        # Load existing state from disk
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                state.last_full_assessment_cycle = data.get('last_full_assessment_cycle', 0)
                state.assessment_interval = data.get(
                    'assessment_interval',
                    self.config.get('full_assessment_interval', 10)
                )

                # Reconstruct benchmarks
                for bid, bdata in data.get('benchmarks', {}).items():
                    state.benchmarks[bid] = Benchmark(
                        id=bdata['id'],
                        name=bdata['name'],
                        description=bdata['description'],
                        category=bdata['category'],
                        source=bdata['source'],
                        target_level=bdata['target_level'],
                        current_level=bdata.get('current_level', 0.0),
                        assessment_history=[
                            tuple(h) for h in bdata.get('assessment_history', [])
                        ]
                    )

                # Reconstruct assessment history
                for adata in data.get('assessment_history', []):
                    state.assessment_history.append(AssessmentResult(
                        cycle_number=adata['cycle_number'],
                        timestamp=adata['timestamp'],
                        benchmarks_assessed=adata.get('benchmarks_assessed', []),
                        overall_score=adata.get('overall_score', 0.0),
                        new_capabilities_desired=adata.get('new_capabilities_desired', []),
                        strengths=adata.get('strengths', []),
                        weaknesses=adata.get('weaknesses', []),
                        trend=adata.get('trend', 'unknown')
                    ))
            except Exception:
                pass

        # Merge config benchmarks (predefined ones always present)
        for bconf in self.config.get('benchmarks', []):
            bid = bconf['id']
            if bid not in state.benchmarks:
                state.benchmarks[bid] = Benchmark(
                    id=bid,
                    name=bconf['name'],
                    description=bconf['description'],
                    category=bconf['category'],
                    source='predefined',
                    target_level=bconf['target_level']
                )
            else:
                # Update target_level and description from config if predefined
                existing = state.benchmarks[bid]
                if existing.source == 'predefined':
                    existing.target_level = bconf['target_level']
                    existing.description = bconf['description']
                    existing.name = bconf['name']

        return state

    def _save_state(self):
        """Save state to file."""
        data = {
            'last_full_assessment_cycle': self.state.last_full_assessment_cycle,
            'assessment_interval': self.state.assessment_interval,
            'benchmarks': {
                bid: {
                    'id': b.id,
                    'name': b.name,
                    'description': b.description,
                    'category': b.category,
                    'source': b.source,
                    'target_level': b.target_level,
                    'current_level': b.current_level,
                    'assessment_history': [list(h) for h in b.assessment_history[-20:]]
                }
                for bid, b in self.state.benchmarks.items()
            },
            'assessment_history': [
                {
                    'cycle_number': a.cycle_number,
                    'timestamp': a.timestamp,
                    'benchmarks_assessed': a.benchmarks_assessed,
                    'overall_score': a.overall_score,
                    'new_capabilities_desired': a.new_capabilities_desired,
                    'strengths': a.strengths,
                    'weaknesses': a.weaknesses,
                    'trend': a.trend
                }
                for a in self.state.assessment_history[-50:]
            ]
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))

    def should_run_full_assessment(self, cycle_number: int) -> bool:
        """Check if it's time for a full assessment."""
        if self.state.assessment_interval <= 0:
            return False
        cycles_since = cycle_number - self.state.last_full_assessment_cycle
        return cycles_since >= self.state.assessment_interval

    def add_agent_benchmark(
        self,
        name: str,
        description: str,
        category: str,
        target_level: float,
        cycle_number: int
    ) -> str:
        """Add an agent-defined benchmark. Returns the benchmark ID."""
        bid = name.lower().replace(' ', '_').replace('-', '_')
        # Avoid collisions
        base_bid = bid
        counter = 1
        while bid in self.state.benchmarks:
            bid = f"{base_bid}_{counter}"
            counter += 1

        self.state.benchmarks[bid] = Benchmark(
            id=bid,
            name=name,
            description=description,
            category=category,
            source='agent_defined',
            target_level=min(1.0, max(0.0, target_level)),
            assessment_history=[(cycle_number, 0.0, "Benchmark created by agent")]
        )
        self._save_state()
        return bid

    def update_benchmark_level(
        self,
        benchmark_id: str,
        level: float,
        notes: str,
        cycle_number: int
    ):
        """Update the current level for a benchmark."""
        if benchmark_id not in self.state.benchmarks:
            return
        b = self.state.benchmarks[benchmark_id]
        b.current_level = min(1.0, max(0.0, level))
        b.assessment_history.append((cycle_number, b.current_level, notes))
        self._save_state()

    def record_full_assessment(self, result: AssessmentResult):
        """Record a full assessment result."""
        # Update benchmark levels from the assessment
        for ba in result.benchmarks_assessed:
            bid = ba.get('id')
            level = ba.get('level', 0.0)
            notes = ba.get('notes', '')
            if bid and bid in self.state.benchmarks:
                b = self.state.benchmarks[bid]
                b.current_level = min(1.0, max(0.0, level))
                b.assessment_history.append(
                    (result.cycle_number, b.current_level, notes)
                )

        # Compute trend
        result.trend = self.get_trend()

        self.state.assessment_history.append(result)
        self.state.last_full_assessment_cycle = result.cycle_number
        self._save_state()

    def get_assessment_summary(self) -> str:
        """Get a lightweight summary for every-cycle context."""
        benchmarks = self.state.benchmarks
        if not benchmarks:
            return "## Capabilities Assessment\nNo benchmarks configured.\n"

        # Overall score
        levels = [b.current_level for b in benchmarks.values()]
        overall = sum(levels) / len(levels) if levels else 0.0

        # Trend
        trend = self.get_trend()

        # Find top strengths and weaknesses by gap from target
        sorted_by_gap = sorted(
            benchmarks.values(),
            key=lambda b: b.current_level - b.target_level
        )
        weaknesses = sorted_by_gap[:2]
        strengths = sorted_by_gap[-2:]

        summary = f"## Capabilities Assessment (Overall: {overall:.2f}, Trend: {trend})\n"

        if any(b.current_level > 0 for b in benchmarks.values()):
            summary += "**Strengths**: "
            summary += ", ".join(
                f"{b.name} ({b.current_level:.2f}/{b.target_level:.2f})"
                for b in strengths if b.current_level > 0
            )
            summary += "\n**Needs work**: "
            summary += ", ".join(
                f"{b.name} ({b.current_level:.2f}/{b.target_level:.2f})"
                for b in weaknesses
            )
            summary += "\n"
        else:
            summary += "No assessments completed yet. "
            summary += f"Next full assessment due.\n"

        last_cycle = self.state.last_full_assessment_cycle
        interval = self.state.assessment_interval
        summary += f"Last full assessment: cycle {last_cycle}, "
        summary += f"interval: every {interval} cycles\n"

        return summary

    def get_full_assessment_prompt(self, cycle_number: int) -> str:
        """Build a detailed prompt for a full assessment cycle."""
        prompt = "\n### CAPABILITIES ASSESSMENT (Full)\n"
        prompt += "This is a full assessment cycle. Rate yourself on each "
        prompt += "benchmark below (0.0 to 1.0 scale).\n\n"

        for bid, b in self.state.benchmarks.items():
            prompt += f"**{b.name}** (id: {bid})\n"
            prompt += f"  Description: {b.description}\n"
            prompt += f"  Category: {b.category}\n"
            prompt += f"  Target: {b.target_level}\n"
            prompt += f"  Current: {b.current_level}\n"

            # Include recent history
            if b.assessment_history:
                recent = b.assessment_history[-3:]
                prompt += "  Recent history:\n"
                for cycle, level, notes in recent:
                    prompt += f"    - Cycle {cycle}: {level:.2f}"
                    if notes:
                        prompt += f" ({notes})"
                    prompt += "\n"
            prompt += "\n"

        prompt += """Respond with a **CAPABILITIES_ASSESSMENT** section containing:
- For each benchmark: `benchmark_id: X.XX - notes`
- OVERALL_SCORE: X.XX
- STRENGTHS: comma-separated list
- WEAKNESSES: comma-separated list
- NEW_CAPABILITIES_DESIRED: any capabilities you wish you had
"""
        return prompt

    def parse_assessment_response(self, content: str, cycle_number: int) -> Optional[AssessmentResult]:
        """Parse the LLM's assessment response into an AssessmentResult."""
        # Look for the CAPABILITIES_ASSESSMENT section
        section_pattern = (
            r'(?:\*\*CAPABILITIES_ASSESSMENT\*\*|#{1,3}\s*CAPABILITIES.?ASSESSMENT)'
            r':?\s*(.*?)(?=\*\*[A-Z]|#{1,3}\s*[A-Z]|\Z)'
        )
        match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return None

        section_text = match.group(1).strip()

        benchmarks_assessed = []
        # Match lines like: benchmark_id: 0.75 - some notes
        # or: benchmark_id: 0.75
        rating_pattern = r'(\w+):\s*(-?[\d.]+)\s*(?:-\s*(.+?))?$'
        for line in section_text.split('\n'):
            line = line.strip().lstrip('- ')
            m = re.match(rating_pattern, line)
            if m:
                bid = m.group(1)
                # Skip known non-benchmark keys
                if bid.upper() in ('OVERALL_SCORE', 'STRENGTHS', 'WEAKNESSES',
                                   'NEW_CAPABILITIES_DESIRED', 'TREND'):
                    continue
                if bid in self.state.benchmarks:
                    benchmarks_assessed.append({
                        'id': bid,
                        'level': min(1.0, max(0.0, float(m.group(2)))),
                        'notes': m.group(3).strip() if m.group(3) else ''
                    })

        # Extract overall score
        overall_match = re.search(r'OVERALL_SCORE:\s*([\d.]+)', section_text, re.IGNORECASE)
        overall_score = float(overall_match.group(1)) if overall_match else 0.0
        overall_score = min(1.0, max(0.0, overall_score))

        # Extract strengths
        strengths = []
        strengths_match = re.search(r'STRENGTHS:\s*(.+?)$', section_text, re.MULTILINE | re.IGNORECASE)
        if strengths_match:
            strengths = [s.strip() for s in strengths_match.group(1).split(',') if s.strip()]

        # Extract weaknesses
        weaknesses = []
        weaknesses_match = re.search(r'WEAKNESSES:\s*(.+?)$', section_text, re.MULTILINE | re.IGNORECASE)
        if weaknesses_match:
            weaknesses = [w.strip() for w in weaknesses_match.group(1).split(',') if w.strip()]

        # Extract new capabilities desired
        new_caps = []
        caps_match = re.search(
            r'NEW_CAPABILITIES_DESIRED:\s*(.+?)$', section_text, re.MULTILINE | re.IGNORECASE
        )
        if caps_match:
            new_caps = [c.strip() for c in caps_match.group(1).split(',') if c.strip()]

        return AssessmentResult(
            cycle_number=cycle_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            benchmarks_assessed=benchmarks_assessed,
            overall_score=overall_score,
            new_capabilities_desired=new_caps,
            strengths=strengths,
            weaknesses=weaknesses,
            trend=self.get_trend()
        )

    def get_trend(self) -> str:
        """Compare last 3 assessments to determine trend."""
        history = self.state.assessment_history
        if len(history) < 2:
            return "unknown"

        recent = history[-3:]
        scores = [a.overall_score for a in recent]

        if len(scores) < 2:
            return "unknown"

        # Check direction of movement
        diffs = [scores[i+1] - scores[i] for i in range(len(scores) - 1)]
        avg_diff = sum(diffs) / len(diffs)

        if avg_diff > 0.05:
            return "improving"
        elif avg_diff < -0.05:
            return "declining"
        else:
            return "stable"
