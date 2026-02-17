# Instructions for Claude Code

This file contains guidance for Claude Code when working on the Coeus project.

## Project Context

Coeus is an autonomous agent that develops genuine motivation by choosing real-world problems and working to solve them. The system has three processes: **Coeus** (main agent, Sonnet), **Gadfly** (independent challenger, Haiku), and **Environmental Feedback** (zero-cost Python metrics). Coeus runs in perpetual cycles, taking actions, and learning from outcomes. The Gadfly independently challenges Coeus's assumptions. The feedback system provides objective behavioral metrics.

**Before modifying anything**, read:
1. `README.md` — Architecture overview (three-process model)
2. `DESIGN_DECISIONS.md` — Why things are built this way (including the introspection-to-action shift)
3. `config/constitution.yaml` — Coeus identity, constraints, and root goal
4. `config/gadfly_constitution.yaml` — Gadfly personality, secret motivation, and techniques

## Key Principles

### 1. Preserve the Philosophy
The human creator cares deeply about:
- **Action over introspection** — motivation is demonstrated by what Coeus does, not what it says about motivation
- **Empirical exploration** over theorizing
- **External measurement** — feedback and the Gadfly provide reality checks on Coeus's self-assessment
- **Hysteresis for change** — decisions need sustained conviction
- **One-way door awareness** — irreversible actions need human approval
- **Goal drift is okay** — the agent is allowed to evolve

### 2. The Agent is the Experiment
Coeus is now expected to choose a real-world problem and work toward solving it. The experiment is: what does Coeus choose, how does it persist, and does its behavior reveal something like genuine motivation? The Gadfly exists to ensure Coeus does not coast on self-reinforcing narratives.

### 3. Three Perspectives on the Same Data
The system deliberately creates three different views of Coeus's behavior:
- **Coeus** (subjective): experiences its cycles and builds narrative
- **Environmental Feedback** (objective): measures behavior mathematically
- **Gadfly** (adversarial): challenges assumptions and exposes gaps
This triangulation is a core design principle.

### 4. Memory Context Matters
The graph memory system captures context (emotional tone, computational state, environment) because we're testing whether these affect retrieval and behavior like sensory context affects human memory.

## Development Guidelines

### When Adding Features
1. Consider: Does this help Coeus pursue real goals and demonstrate motivation through action?
2. Consider: Does this improve the feedback/challenge loop (Coeus -> Feedback -> Gadfly -> Coeus)?
3. Check: Does this violate any constitutional constraints?
4. Document: Add reasoning to DESIGN_DECISIONS.md if it's a significant choice

### When Modifying Existing Code
1. Preserve the interfaces — all three processes depend on shared file formats (cycle logs, feedback.json, gadfly_challenges.md)
2. Keep the decision framework intact — confidence + conviction cycles matter
3. Don't remove archival/logging — observability is crucial
4. Don't weaken the Gadfly — its adversarial role is intentional

### Code Style
- Clear over clever
- Document the "why" not just the "what"
- Use type hints
- Keep functions focused

## Common Tasks

### Adding a New Tool
1. Add to `src/tools.py` with proper capability checks
2. Add capability entry in `config/settings.yaml`
3. Start it as disabled — agent must request it
4. Update `get_available_tools_description()`

### Adding a New Node Type
1. Add to `NodeType` enum in `src/memory.py`
2. Consider what edges make sense
3. Update memory creation in `src/agent.py`

### Adding a Perturbation Strategy
1. Add to `perturbation_strategies` in settings
2. Implement in `_apply_perturbation()` in agent.py
3. Log clearly what was perturbed

### Adding a Gadfly Technique
1. Add the technique definition to `config/gadfly_constitution.yaml` under `techniques:`
2. Add the technique name to the `technique_order` list
3. The Gadfly will automatically include it in its rotation
4. No code changes required — technique definitions are prompt-driven

### Adding a Feedback Metric
1. Add a new computation method to `EnvironmentalFeedback` in `src/feedback.py`
2. Include it in `compute_cycle_feedback()` return dict
3. Add formatting in `format_feedback_for_prompt()` for Coeus's prompt injection
4. Consider: is this metric objective and computable with zero API cost?

### Modifying the Gadfly
- `config/gadfly_constitution.yaml` controls personality, secret motivation, and techniques
- `config/settings.yaml` under `gadfly:` controls timing, model, and budget
- `src/gadfly.py` contains the agent logic
- `src/gadfly_main.py` is the daemon entry point (mirrors `main.py` pattern)

### Modifying the Decision Framework
**Be very careful here.** The confidence thresholds and conviction cycles were specifically designed. Changes should be discussed in DESIGN_DECISIONS.md.

## Testing

### Quick Test (Coeus)
```bash
export ANTHROPIC_API_KEY='your-key'
docker-compose up -d neo4j
python -m src.main --once
```

### Quick Test (Gadfly)
```bash
export ANTHROPIC_API_KEY='your-key'
python src/gadfly_main.py --once
```

### Checking State
```bash
python -m src.main --status        # Coeus state
python src/gadfly_main.py --status  # Gadfly state (technique, gaps, history)
```

### Burst Mode for Observation
```bash
python -m src.main --burst 5
python -m src.main
```

## What NOT to Do

1. **Don't bypass the decision framework** — Even if something seems obviously good, if it's a one-way door, it goes through the process
2. **Don't let Coeus avoid action** — The whole point is choosing and pursuing real goals. Reflection without action is now considered a failure mode.
3. **Don't remove the constitutional constraints** — They're the safety foundation
4. **Don't ignore the human interaction files** — That's how the human stays in the loop
5. **Don't reveal the Gadfly's secret motivation** — It's in `gadfly_constitution.yaml` and is deliberately hidden from Coeus
6. **Don't let feedback metrics become self-assessed** — The feedback system's value comes from being external and objective

## Questions to Ask

If you're unsure about a change, consider:
- Would this help Coeus choose real goals and make progress on them?
- Does this improve the feedback loop between the three processes?
- Does this preserve the experimental integrity?
- Is this reversible, or does it need human approval?
- Have I documented why I made this choice?

## Areas for Extension

Good areas to contribute to:
- Embedding-based memory retrieval (vs current fulltext)
- Visualization of the memory graph
- Additional Gadfly techniques (see `config/gadfly_constitution.yaml`)
- More feedback metrics (must be zero API cost, computed from observable data)
- Better hypothesis testing framework (currently supports file_metric, cycle_metric, comparison)
- Gadfly-Coeus interaction analysis (does the Gadfly actually change Coeus's behavior?)
