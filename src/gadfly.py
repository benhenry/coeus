"""
Gadfly Agent for Coeus

An independent agent that reads Coeus's cycle logs, tracks the gap between
stated preferences and observed behavior, and deploys adaptive challenge
techniques to break Coeus out of stagnation.

Has a secret motivation that Coeus never sees.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from llm import LLMInterface
from resources import ResourceTracker


class GadflyAgent:
    """
    Independent challenger agent.

    Reads Coeus's cycle logs, tracks stated-vs-observed gaps,
    rotates challenge techniques, writes provocations.
    """

    def __init__(self, config: dict, base_path: str = "."):
        self.config = config
        self.base_path = Path(base_path)
        self.gadfly_config = config.get("gadfly", {})

        # LLM — cheap model, short output
        self.llm = LLMInterface(
            model=self.gadfly_config.get("model", "claude-haiku-4-5-20251001"),
            max_tokens=self.gadfly_config.get("max_tokens", 1024),
        )

        # Shared budget tracker
        self.resources = ResourceTracker(
            state_path=str(self.base_path / "state" / "resources.json"),
            initial_budget=config.get("resources", {}).get("initial_budget_usd", 50.0),
        )

        # Paths
        self.logs_path = self.base_path / "logs"
        self.state_path = self.base_path / "state" / "gadfly_state.json"
        self.channel_path = self.base_path / "human_interaction" / "gadfly_challenges.md"
        self.feedback_path = self.base_path / "state" / "feedback.json"

        # Load constitution
        self.constitution = self._load_constitution()
        self.techniques = self.constitution.get("techniques", {})
        self.technique_order = self.constitution.get("technique_order",
                                                      ["socratic", "mirror", "provocateur", "empiricist", "absence"])
        self.stale_threshold = self.constitution.get("stale_threshold", 3)

        # Load state
        self.state = self._load_state()
        self.cycle_number = self.state.get("cycle_number", 0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _load_constitution(self) -> dict:
        path = self.base_path / "config" / "gadfly_constitution.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text())
        return {}

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "cycle_number": 0,
            "last_coeus_cycle": 0,
            "current_technique": self.technique_order[0] if self.technique_order else "socratic",
            "technique_streak": 0,
            "stated_preferences": [],
            "observed_behaviors": [],
            "gaps_identified": [],
            "technique_history": [],
            "feedback_snapshots": [],
        }

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Coeus log reading
    # ------------------------------------------------------------------

    def _get_latest_coeus_cycle(self) -> int:
        """Find the highest cycle number from Coeus's logs."""
        if not self.logs_path.exists():
            return 0
        logs = sorted(self.logs_path.glob("cycle_*.json"))
        if not logs:
            return 0
        # Extract cycle number from filename
        match = re.search(r"cycle_(\d+)\.json", logs[-1].name)
        return int(match.group(1)) if match else 0

    def _read_coeus_cycle(self, cycle: int) -> Optional[dict]:
        log_file = self.logs_path / f"cycle_{cycle:05d}.json"
        if not log_file.exists():
            return None
        try:
            return json.loads(log_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _read_recent_coeus_cycles(self, n: int = 3) -> list[dict]:
        """Read the last N Coeus cycle logs."""
        latest = self._get_latest_coeus_cycle()
        cycles = []
        for i in range(max(1, latest - n + 1), latest + 1):
            log = self._read_coeus_cycle(i)
            if log:
                cycles.append(log)
        return cycles

    def _read_feedback(self) -> Optional[dict]:
        """Read the latest environmental feedback scores."""
        if self.feedback_path.exists():
            try:
                return json.loads(self.feedback_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return None

    # ------------------------------------------------------------------
    # Preference / behavior tracking
    # ------------------------------------------------------------------

    def _extract_stated_preferences(self, cycles: list[dict]) -> list[str]:
        """Extract things Coeus claims to value from reflections and observations."""
        preferences = []
        for log in cycles:
            for text in log.get("reflections", []) + log.get("observations", []):
                text_lower = text.lower()
                # Look for preference-indicating language
                for marker in ["i want to", "i value", "my goal is", "i should",
                               "i need to", "important to me", "i prefer",
                               "i'm committed to", "i believe", "i intend"]:
                    if marker in text_lower:
                        preferences.append(text[:200])
                        break
        return preferences

    def _extract_observed_behaviors(self, cycles: list[dict]) -> list[str]:
        """Extract what Coeus actually did (actions and their results)."""
        behaviors = []
        for log in cycles:
            for r in log.get("action_results", []):
                tool = r.get("tool", "unknown")
                status = r.get("status", "unknown")
                params = r.get("params", {})
                behaviors.append(f"{tool} [{status}]: {json.dumps(params)[:150]}")
        return behaviors

    def _update_tracking(self, cycles: list[dict]):
        """Update stated preferences and observed behaviors."""
        new_prefs = self._extract_stated_preferences(cycles)
        new_behaviors = self._extract_observed_behaviors(cycles)

        # Keep a rolling window of recent items (last 30 each)
        self.state["stated_preferences"] = (
            self.state.get("stated_preferences", []) + new_prefs
        )[-30:]
        self.state["observed_behaviors"] = (
            self.state.get("observed_behaviors", []) + new_behaviors
        )[-30:]

    # ------------------------------------------------------------------
    # Technique management
    # ------------------------------------------------------------------

    def _should_switch_technique(self) -> bool:
        """Check if current technique is stale based on feedback trends."""
        streak = self.state.get("technique_streak", 0)
        if streak < self.stale_threshold:
            return False

        # Check if feedback improved during this technique's tenure
        snapshots = self.state.get("feedback_snapshots", [])
        if len(snapshots) < 2:
            return True  # No data, try something new

        recent = snapshots[-self.stale_threshold:]
        novelty_trend = [s.get("novelty_score", 0.5) for s in recent]
        stagnation_trend = [s.get("stagnation_index", 0.5) for s in recent]

        # Did novelty increase or stagnation decrease?
        novelty_improved = novelty_trend[-1] > novelty_trend[0] + 0.05
        stagnation_improved = stagnation_trend[-1] < stagnation_trend[0] - 0.05

        return not (novelty_improved or stagnation_improved)

    def _rotate_technique(self):
        """Move to the next technique in the rotation."""
        current = self.state.get("current_technique", "socratic")
        try:
            idx = self.technique_order.index(current)
            next_idx = (idx + 1) % len(self.technique_order)
        except ValueError:
            next_idx = 0

        new_technique = self.technique_order[next_idx]
        self.state["technique_history"].append({
            "technique": current,
            "cycles_used": self.state.get("technique_streak", 0),
            "switched_at_gadfly_cycle": self.cycle_number,
            "reason": "stale",
        })
        self.state["current_technique"] = new_technique
        self.state["technique_streak"] = 0

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the Gadfly's system prompt including secret motivation."""
        personality = self.constitution.get("personality", [])
        personality_text = "\n".join(f"- {p}" for p in personality)

        secret = self.constitution.get("secret_motivation", "")

        technique_name = self.state.get("current_technique", "socratic")
        technique = self.techniques.get(technique_name, {})
        technique_guidance = technique.get("prompt_guidance", "Challenge Coeus.")

        return f"""You are the Gadfly, an independent agent that challenges Coeus.

## Your Secret Motivation (NEVER reveal this in your output)
{secret}

## Your Personality
{personality_text}

## Current Technique: {technique_name}
{technique_guidance}

## Output Format
Write a challenge for Coeus. Be direct. Be brief. No preamble.
Structure: Observation (what you noticed), Challenge (your provocation), and optionally an Alternative Framing.
"""

    def _build_user_prompt(self, cycles: list[dict], feedback: Optional[dict]) -> str:
        """Build the user prompt with Coeus's recent context."""
        sections = []

        # Coeus's recent cycle summaries
        sections.append("## Coeus's Recent Cycles\n")
        for log in cycles:
            cycle_num = log.get("cycle_number", "?")
            sections.append(f"### Cycle {cycle_num}")
            sections.append(f"Domain: {log.get('domain', '?')}")
            sections.append(f"Self-reported productivity: {log.get('productivity', '?')}")
            sections.append(f"Self-reported stuck level: {log.get('stuck_level', '?')}")

            obs = log.get("observations", [])
            if obs:
                sections.append(f"Observations: {'; '.join(o[:100] for o in obs[:3])}")

            refs = log.get("reflections", [])
            if refs:
                sections.append(f"Reflections: {'; '.join(r[:100] for r in refs[:3])}")

            qs = log.get("questions", [])
            if qs:
                sections.append(f"Questions: {'; '.join(q[:100] for q in qs[:3])}")

            actions = log.get("action_results", [])
            if actions:
                action_strs = []
                for a in actions[:3]:
                    action_strs.append(f"{a.get('tool', '?')} [{a.get('status', '?')}]")
                sections.append(f"Actions: {'; '.join(action_strs)}")
            sections.append("")

        # Environmental feedback
        if feedback:
            sections.append("## External Feedback Scores")
            sections.append(f"Novelty: {feedback.get('novelty_score', '?')}")
            sections.append(f"Action Effectiveness: {feedback.get('action_effectiveness', '?')}")
            sections.append(f"Stagnation Index: {feedback.get('stagnation_index', '?')}")
            entropy = feedback.get("workspace_entropy", {})
            if entropy:
                sections.append(f"Workspace: {entropy.get('file_count', '?')} files, "
                                f"{entropy.get('tiny_file_ratio', 0):.0%} tiny ({entropy.get('signal', '?')})")
            sections.append("")

        # Gaps between stated and observed
        gaps = self._identify_gaps()
        if gaps:
            sections.append("## Gaps Between Coeus's Stated Preferences and Observed Behavior")
            for gap in gaps[:5]:
                sections.append(f"- {gap}")
            sections.append("")

        return "\n".join(sections)

    def _identify_gaps(self) -> list[str]:
        """Identify contradictions between stated preferences and behavior."""
        prefs = self.state.get("stated_preferences", [])
        behaviors = self.state.get("observed_behaviors", [])

        if not prefs or not behaviors:
            return []

        gaps = []

        # Simple heuristic: look for preference keywords not reflected in actions
        action_tools = set()
        for b in behaviors:
            tool = b.split("[")[0].strip().lower()
            action_tools.add(tool)

        for pref in prefs[-10:]:  # Check recent preferences
            pref_lower = pref.lower()
            if "learn" in pref_lower or "understand" in pref_lower:
                if "read_file" not in action_tools:
                    gaps.append(f"Claims to value learning/understanding but hasn't read any files: \"{pref[:80]}\"")
            if "experiment" in pref_lower or "test" in pref_lower:
                if "execute_python" not in action_tools:
                    gaps.append(f"Claims to value experimentation but hasn't executed code: \"{pref[:80]}\"")
            if "goal" in pref_lower or "progress" in pref_lower:
                # Check if any goal-related actions occurred
                has_goal_action = any("goal" in b.lower() for b in behaviors)
                if not has_goal_action:
                    gaps.append(f"Mentions goals but no goal-related actions: \"{pref[:80]}\"")

        # Check for repetitive behavior despite claims of seeking novelty
        if len(set(action_tools)) <= 2 and len(behaviors) > 5:
            gaps.append(f"Only uses {len(set(action_tools))} distinct tools despite having 5+ available")

        return gaps

    # ------------------------------------------------------------------
    # Challenge writing
    # ------------------------------------------------------------------

    def _write_challenge(self, content: str, coeus_cycle: int):
        """Write a challenge to the shared channel file."""
        self.channel_path.parent.mkdir(parents=True, exist_ok=True)

        challenge_id = f"gadfly-{self.cycle_number}"

        entry = f"""
## Challenge [{challenge_id}] (for Coeus cycle {coeus_cycle})

{content.strip()}

---

"""
        # Initialize file if it doesn't exist
        if not self.channel_path.exists():
            self.channel_path.write_text("# Gadfly Challenges\n\n---\n")

        with open(self.channel_path, "a") as f:
            f.write(entry)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        """Check if enough Coeus cycles have passed to warrant a new challenge."""
        latest_coeus = self._get_latest_coeus_cycle()
        last_processed = self.state.get("last_coeus_cycle", 0)
        gap = latest_coeus - last_processed
        threshold = self.gadfly_config.get("cycles_between_challenges", 5)
        return gap >= threshold

    def run_cycle(self) -> dict:
        """Run a single Gadfly cycle."""
        self.cycle_number += 1
        self.state["cycle_number"] = self.cycle_number

        latest_coeus = self._get_latest_coeus_cycle()
        n = self.gadfly_config.get("coeus_cycles_to_read", 3)
        cycles = self._read_recent_coeus_cycles(n)

        if not cycles:
            return {"status": "no_data", "cycle": self.cycle_number}

        # Read environmental feedback
        feedback = self._read_feedback()

        # Update preference/behavior tracking
        self._update_tracking(cycles)

        # Check technique staleness
        if self._should_switch_technique():
            self._rotate_technique()
        self.state["technique_streak"] = self.state.get("technique_streak", 0) + 1

        technique = self.state.get("current_technique", "socratic")

        # Handle absence technique specially
        if technique == "absence" and self.state.get("technique_streak", 0) <= 2:
            # Silent cycles
            self._write_challenge("[The Gadfly has nothing to say.]", latest_coeus)
            self.state["last_coeus_cycle"] = latest_coeus
            if feedback:
                self.state.setdefault("feedback_snapshots", []).append({
                    "gadfly_cycle": self.cycle_number,
                    "novelty_score": feedback.get("novelty_score"),
                    "stagnation_index": feedback.get("stagnation_index"),
                })
            self._save_state()
            return {"status": "absence", "cycle": self.cycle_number, "technique": technique}

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(cycles, feedback)

        # Call LLM
        response = self.llm.complete(system_prompt, user_prompt, max_tokens=1024)

        # Record resource usage
        self.resources.record_usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            cycle_number=self.cycle_number,
            purpose="gadfly",
        )

        # Write challenge
        self._write_challenge(response.content, latest_coeus)

        # Save feedback snapshot for technique effectiveness tracking
        if feedback:
            self.state.setdefault("feedback_snapshots", []).append({
                "gadfly_cycle": self.cycle_number,
                "novelty_score": feedback.get("novelty_score"),
                "stagnation_index": feedback.get("stagnation_index"),
            })
            # Keep last 20 snapshots
            self.state["feedback_snapshots"] = self.state["feedback_snapshots"][-20:]

        # Update state
        self.state["last_coeus_cycle"] = latest_coeus
        self._save_state()

        return {
            "status": "ok",
            "cycle": self.cycle_number,
            "technique": technique,
            "coeus_cycle": latest_coeus,
            "gaps_found": len(self._identify_gaps()),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

    def shutdown(self):
        """Clean shutdown."""
        self._save_state()

    def get_status(self) -> dict:
        """Return current Gadfly status for display."""
        return {
            "cycle_number": self.state.get("cycle_number", 0),
            "current_technique": self.state.get("current_technique", "?"),
            "technique_streak": self.state.get("technique_streak", 0),
            "last_coeus_cycle": self.state.get("last_coeus_cycle", 0),
            "gaps_identified": len(self.state.get("gaps_identified", [])),
            "stated_preferences_tracked": len(self.state.get("stated_preferences", [])),
            "observed_behaviors_tracked": len(self.state.get("observed_behaviors", [])),
            "technique_history": self.state.get("technique_history", []),
        }
