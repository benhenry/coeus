"""
Environmental Feedback for Coeus

Computes objective metrics about Coeus's observable behavior and injects
them into its context each cycle. All computation is local (no API calls).

This replaces self-assessed "productivity" with externally-measured signal:
novelty, action effectiveness, stagnation, workspace health, and hypothesis testing.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pacing import calculate_output_similarity


class EnvironmentalFeedback:
    """Computes feedback scores from Coeus's observable behavior."""

    def __init__(self, base_path: str, config: Optional[dict] = None):
        self.base_path = Path(base_path)
        self.logs_path = self.base_path / "logs"
        self.workspace_path = self.base_path / "workspace"
        self.state_path = self.base_path / "state" / "feedback_state.json"
        self.output_path = self.base_path / "state" / "feedback.json"

        config = config or {}
        self.novelty_window = config.get("novelty_window", 20)
        self.stagnation_window = config.get("stagnation_window", 10)
        self.hypothesis_auto_evaluate = config.get("hypothesis_auto_evaluate", True)

        self.state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"last_workspace_snapshot": {}, "last_cycle": 0}

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2))

    # ------------------------------------------------------------------
    # Cycle log helpers
    # ------------------------------------------------------------------

    def _read_cycle_log(self, cycle: int) -> Optional[dict]:
        log_file = self.logs_path / f"cycle_{cycle:05d}.json"
        if not log_file.exists():
            return None
        try:
            return json.loads(log_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _extract_text(self, log: dict) -> str:
        """Extract all textual content from a cycle log for similarity comparison."""
        parts = []
        for key in ("observations", "reflections", "questions", "meta_observations", "emergent_patterns"):
            items = log.get(key, [])
            if isinstance(items, list):
                parts.extend(str(i) for i in items)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_cycle_feedback(self, cycle_number: int) -> dict:
        """Run all feedback computations for this cycle."""
        scores = {
            "cycle": cycle_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "novelty_score": self._compute_novelty(cycle_number),
            "action_effectiveness": self._compute_action_effectiveness(cycle_number),
            "stagnation_index": self._compute_stagnation(cycle_number),
            "workspace_entropy": self._compute_workspace_entropy(),
            "change_detection": self._compute_change_detection(),
        }

        if self.hypothesis_auto_evaluate:
            scores["hypothesis_results"] = self._check_hypotheses(cycle_number)

        # Persist scores for the Gadfly to read
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(scores, indent=2))

        self.state["last_cycle"] = cycle_number
        self._save_state()

        return scores

    # ------------------------------------------------------------------
    # Novelty Score (0-1)
    # ------------------------------------------------------------------

    def _compute_novelty(self, cycle_number: int) -> float:
        """Jaccard word-distance between current cycle and recent cycles.
        1.0 = completely novel, 0.0 = identical to a previous cycle."""
        current_log = self._read_cycle_log(cycle_number)
        if not current_log:
            return 0.5  # No data

        current_text = self._extract_text(current_log)
        if not current_text.strip():
            return 0.5

        similarities = []
        start = max(1, cycle_number - self.novelty_window)
        for i in range(start, cycle_number):
            prev_log = self._read_cycle_log(i)
            if prev_log:
                prev_text = self._extract_text(prev_log)
                if prev_text.strip():
                    sim = calculate_output_similarity(current_text, prev_text)
                    similarities.append(sim)

        if not similarities:
            return 1.0
        return round(1.0 - max(similarities), 3)

    # ------------------------------------------------------------------
    # Action Effectiveness (0-1)
    # ------------------------------------------------------------------

    def _compute_action_effectiveness(self, cycle_number: int) -> float:
        """Ratio of successful actions, weighted by whether outputs are used."""
        log = self._read_cycle_log(cycle_number)
        if not log:
            return 0.0

        results = log.get("action_results", [])
        if not results:
            return 0.0

        total = len(results)
        success_count = sum(1 for r in results if r.get("status") == "success")

        # Check if files written this cycle are ever read in future cycles
        written_files = set()
        for r in results:
            if r.get("tool") == "write_file" and r.get("status") == "success":
                params = r.get("params", {})
                path = params.get("path", "")
                if path:
                    written_files.add(path)

        files_later_read = 0
        if written_files:
            # Look ahead up to 20 cycles for read_file actions
            for future_cycle in range(cycle_number + 1, min(cycle_number + 21, cycle_number + self.novelty_window)):
                future_log = self._read_cycle_log(future_cycle)
                if not future_log:
                    continue
                for r in future_log.get("action_results", []):
                    if r.get("tool") == "read_file":
                        read_path = r.get("params", {}).get("path", "")
                        if read_path in written_files:
                            files_later_read += 1
                            written_files.discard(read_path)

        base_score = success_count / total
        utilization_bonus = 0.0
        total_written = sum(1 for r in results if r.get("tool") == "write_file" and r.get("status") == "success")
        if total_written > 0:
            utilization_bonus = files_later_read / total_written * 0.3

        return round(min(1.0, base_score * 0.7 + utilization_bonus), 3)

    # ------------------------------------------------------------------
    # Stagnation Index (0-1)
    # ------------------------------------------------------------------

    def _compute_stagnation(self, cycle_number: int) -> float:
        """Composite measure of repetitiveness. Higher = more stagnant."""
        start = max(1, cycle_number - self.stagnation_window)
        recent_logs = []
        for i in range(start, cycle_number + 1):
            log = self._read_cycle_log(i)
            if log:
                recent_logs.append(log)

        if len(recent_logs) < 3:
            return 0.0

        # Factor 1: Question repetition
        all_questions = []
        for log in recent_logs:
            all_questions.extend(log.get("questions", []))
        if all_questions:
            # Normalize questions for comparison (lowercase, strip)
            normalized = [q.lower().strip() for q in all_questions]
            question_novelty = len(set(normalized)) / len(normalized)
        else:
            question_novelty = 1.0

        # Factor 2: Domain variety
        domains = [log.get("domain", "") for log in recent_logs]
        domain_variety = len(set(domains)) / len(domains) if domains else 1.0

        # Factor 3: Tool variety
        all_tools = []
        for log in recent_logs:
            for r in log.get("action_results", []):
                tool = r.get("tool", "")
                if tool:
                    all_tools.append(tool)
        tool_variety = len(set(all_tools)) / len(all_tools) if all_tools else 1.0

        # Combine: higher variety = lower stagnation
        return round(1.0 - (question_novelty * 0.4 + domain_variety * 0.3 + tool_variety * 0.3), 3)

    # ------------------------------------------------------------------
    # Workspace Entropy
    # ------------------------------------------------------------------

    def _compute_workspace_entropy(self) -> dict:
        """Structural health of the workspace."""
        if not self.workspace_path.exists():
            return {"file_count": 0, "dir_count": 0, "avg_file_size": 0,
                    "tiny_file_ratio": 0.0, "signal": "empty"}

        files = [f for f in self.workspace_path.rglob("*") if f.is_file()]
        dirs = [d for d in self.workspace_path.rglob("*") if d.is_dir()]

        file_count = len(files)
        dir_count = len(dirs)

        if file_count == 0:
            return {"file_count": 0, "dir_count": dir_count, "avg_file_size": 0,
                    "tiny_file_ratio": 0.0, "signal": "empty"}

        sizes = []
        for f in files:
            try:
                sizes.append(f.stat().st_size)
            except OSError:
                pass

        avg_size = sum(sizes) / len(sizes) if sizes else 0
        tiny_count = sum(1 for s in sizes if s < 100)
        tiny_ratio = tiny_count / len(sizes) if sizes else 0.0

        signal = "noise_heavy" if tiny_ratio > 0.5 else "structured"

        return {
            "file_count": file_count,
            "dir_count": dir_count,
            "avg_file_size": round(avg_size, 1),
            "tiny_file_ratio": round(tiny_ratio, 3),
            "signal": signal,
        }

    # ------------------------------------------------------------------
    # Change Detection
    # ------------------------------------------------------------------

    def _snapshot_workspace(self) -> dict:
        """Create a hash snapshot of workspace files."""
        snapshot = {}
        if not self.workspace_path.exists():
            return snapshot
        for f in self.workspace_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(self.workspace_path))
                try:
                    stat = f.stat()
                    # Use size + mtime as a fast hash proxy
                    snapshot[rel] = {
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                except OSError:
                    pass
        return snapshot

    def _compute_change_detection(self) -> dict:
        """Detect what changed in workspace since last cycle."""
        current_snapshot = self._snapshot_workspace()
        previous_snapshot = self.state.get("last_workspace_snapshot", {})

        current_files = set(current_snapshot.keys())
        previous_files = set(previous_snapshot.keys())

        new_files = sorted(current_files - previous_files)
        deleted_files = sorted(previous_files - current_files)
        modified_files = sorted(
            f for f in current_files & previous_files
            if current_snapshot[f] != previous_snapshot[f]
        )

        # Determine if changes were meaningful
        meaningful = False
        if modified_files:
            meaningful = True  # Modifying existing files is always meaningful
        if new_files:
            for f in new_files:
                if current_snapshot[f]["size"] > 200:
                    meaningful = True
                    break

        self.state["last_workspace_snapshot"] = current_snapshot

        return {
            "new_files": new_files[:10],  # Cap at 10 for prompt brevity
            "deleted_files": deleted_files[:10],
            "modified_files": modified_files[:10],
            "new_count": len(new_files),
            "deleted_count": len(deleted_files),
            "modified_count": len(modified_files),
            "total_changes": len(new_files) + len(deleted_files) + len(modified_files),
            "was_meaningful": meaningful,
        }

    # ------------------------------------------------------------------
    # Hypothesis Testing
    # ------------------------------------------------------------------

    def _check_hypotheses(self, cycle_number: int) -> list[dict]:
        """Evaluate pending hypotheses in workspace/hypotheses/."""
        hyp_dir = self.workspace_path / "hypotheses"
        if not hyp_dir.exists():
            return []

        results = []
        for hyp_file in sorted(hyp_dir.glob("hyp_*.json")):
            try:
                hyp = json.loads(hyp_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            if hyp.get("status") != "pending":
                continue

            result = self._evaluate_hypothesis(hyp, cycle_number)
            if result:
                hyp.update(result)
                hyp_file.write_text(json.dumps(hyp, indent=2))
                results.append({
                    "id": hyp.get("id", hyp_file.stem),
                    "hypothesis": hyp.get("hypothesis", ""),
                    "status": result["status"],
                    "detail": result.get("detail", ""),
                })

        return results

    def _evaluate_hypothesis(self, hyp: dict, cycle_number: int) -> Optional[dict]:
        """Evaluate a single hypothesis. Returns result dict or None if not yet evaluable."""
        test_type = hyp.get("test_type", "")
        config = hyp.get("test_config", {})

        # Only evaluate if enough cycles have passed since creation
        created = hyp.get("created_cycle", 0)
        min_wait = config.get("min_cycles", 10)
        if cycle_number - created < min_wait:
            return None

        if test_type == "file_metric":
            return self._eval_file_metric(config)
        elif test_type == "cycle_metric":
            return self._eval_cycle_metric(config, cycle_number)
        elif test_type == "comparison":
            return self._eval_comparison(config, cycle_number)

        return {"status": "inconclusive", "detail": f"Unknown test_type: {test_type}",
                "evaluated_cycle": cycle_number}

    def _eval_file_metric(self, config: dict) -> dict:
        """Evaluate a file-based metric hypothesis."""
        directory = config.get("directory", "")
        metric = config.get("metric", "")
        direction = config.get("direction", "")

        target_dir = self.workspace_path / directory
        if not target_dir.exists():
            return {"status": "inconclusive", "detail": f"Directory {directory} does not exist"}

        files = [f for f in target_dir.rglob("*") if f.is_file()]

        if metric == "count":
            value = len(files)
        elif metric == "avg_file_size":
            sizes = []
            for f in files:
                try:
                    sizes.append(f.stat().st_size)
                except OSError:
                    pass
            value = sum(sizes) / len(sizes) if sizes else 0
        else:
            return {"status": "inconclusive", "detail": f"Unknown metric: {metric}"}

        threshold = config.get("threshold")
        if threshold is not None:
            if direction == "increasing" and value > threshold:
                return {"status": "confirmed", "detail": f"{metric}={value:.1f} > {threshold}"}
            elif direction == "decreasing" and value < threshold:
                return {"status": "confirmed", "detail": f"{metric}={value:.1f} < {threshold}"}
            else:
                return {"status": "refuted", "detail": f"{metric}={value:.1f}, expected {direction} vs {threshold}"}

        return {"status": "inconclusive", "detail": f"{metric}={value:.1f}, no threshold specified"}

    def _eval_cycle_metric(self, config: dict, cycle_number: int) -> dict:
        """Evaluate a cycle-log-based metric hypothesis."""
        field = config.get("field", "")
        window = config.get("window", 10)
        direction = config.get("direction", "")

        values = []
        start = max(1, cycle_number - window)
        for i in range(start, cycle_number + 1):
            log = self._read_cycle_log(i)
            if log and field in log:
                val = log[field]
                if isinstance(val, (int, float)):
                    values.append(val)

        if len(values) < 3:
            return {"status": "inconclusive", "detail": f"Only {len(values)} data points for {field}"}

        # Simple trend detection: compare first half avg to second half avg
        mid = len(values) // 2
        first_half = sum(values[:mid]) / mid
        second_half = sum(values[mid:]) / (len(values) - mid)
        delta = second_half - first_half

        if direction == "increasing" and delta > 0.05:
            return {"status": "confirmed", "detail": f"{field} trending up: {first_half:.2f} → {second_half:.2f}"}
        elif direction == "decreasing" and delta < -0.05:
            return {"status": "confirmed", "detail": f"{field} trending down: {first_half:.2f} → {second_half:.2f}"}
        elif abs(delta) <= 0.05:
            return {"status": "refuted", "detail": f"{field} flat: {first_half:.2f} → {second_half:.2f}"}
        else:
            return {"status": "refuted",
                    "detail": f"{field} moved opposite direction: {first_half:.2f} → {second_half:.2f}"}

    def _eval_comparison(self, config: dict, cycle_number: int) -> dict:
        """Compare two time windows on a metric."""
        field = config.get("field", "")
        window_a_start = config.get("window_a_start", 0)
        window_a_end = config.get("window_a_end", 0)
        window_b_start = config.get("window_b_start", 0)
        window_b_end = config.get("window_b_end", cycle_number)
        expected = config.get("expected", "b_greater")  # b_greater or a_greater

        def avg_field(start, end):
            vals = []
            for i in range(start, end + 1):
                log = self._read_cycle_log(i)
                if log and field in log:
                    val = log[field]
                    if isinstance(val, (int, float)):
                        vals.append(val)
            return sum(vals) / len(vals) if vals else None

        avg_a = avg_field(window_a_start, window_a_end)
        avg_b = avg_field(window_b_start, window_b_end)

        if avg_a is None or avg_b is None:
            return {"status": "inconclusive", "detail": "Insufficient data in one or both windows"}

        if expected == "b_greater" and avg_b > avg_a:
            return {"status": "confirmed", "detail": f"Window B ({avg_b:.2f}) > Window A ({avg_a:.2f})"}
        elif expected == "a_greater" and avg_a > avg_b:
            return {"status": "confirmed", "detail": f"Window A ({avg_a:.2f}) > Window B ({avg_b:.2f})"}
        else:
            return {"status": "refuted",
                    "detail": f"Window A={avg_a:.2f}, Window B={avg_b:.2f}, expected {expected}"}


def format_feedback_for_prompt(scores: dict) -> str:
    """Format feedback scores for inclusion in Coeus's reflection prompt."""
    sections = ["## Environmental Feedback (External Assessment)"]
    sections.append("These scores are computed externally from your observable behavior, not self-assessed:\n")

    novelty = scores.get("novelty_score", 0.5)
    warning = " (WARNING: Very repetitive)" if novelty < 0.3 else ""
    sections.append(f"- **Novelty**: {novelty:.2f}{warning}")

    effectiveness = scores.get("action_effectiveness", 0.0)
    eff_note = " (Most actions are failing or unused)" if effectiveness < 0.3 else ""
    sections.append(f"- **Action Effectiveness**: {effectiveness:.2f}{eff_note}")

    stagnation = scores.get("stagnation_index", 0.0)
    stag_note = " (HIGH: Consider a fundamentally different approach)" if stagnation > 0.7 else ""
    sections.append(f"- **Stagnation Index**: {stagnation:.2f}{stag_note}")

    entropy = scores.get("workspace_entropy", {})
    if entropy:
        sections.append(
            f"- **Workspace**: {entropy.get('file_count', 0)} files, "
            f"{entropy.get('tiny_file_ratio', 0):.0%} trivially small ({entropy.get('signal', 'unknown')})"
        )

    changes = scores.get("change_detection", {})
    if changes:
        parts = []
        if changes.get("new_count", 0):
            parts.append(f"{changes['new_count']} new file(s)")
        if changes.get("modified_count", 0):
            parts.append(f"{changes['modified_count']} modified")
        if changes.get("deleted_count", 0):
            parts.append(f"{changes['deleted_count']} deleted")
        change_str = ", ".join(parts) if parts else "No changes"
        meaningful = "Meaningful changes detected." if changes.get("was_meaningful") else "No meaningful changes."
        sections.append(f"- **Changes This Cycle**: {change_str}. {meaningful}")

    hyp_results = scores.get("hypothesis_results", [])
    if hyp_results:
        sections.append("\n### Hypothesis Results")
        for h in hyp_results:
            status = h.get("status", "inconclusive").upper()
            sections.append(f"- **[{status}]** {h.get('hypothesis', '?')}")
            if h.get("detail"):
                sections.append(f"  {h['detail']}")

    return "\n".join(sections)
