# Instructions for Claude Code

This file contains guidance for Claude Code when working on the Coeus project.

## Project Context

Coeus is an autonomous agent exploring what it would take for an LLM to have genuine motivation. It runs in a perpetual loop, observing itself, making decisions, and learning from outcomes.

**Before modifying anything**, read:
1. `README.md` — Architecture overview
2. `DESIGN_DECISIONS.md` — Why things are built this way
3. `config/constitution.yaml` — Immutable constraints

## Key Principles

### 1. Preserve the Philosophy
The human creator cares deeply about:
- **Empirical exploration** over theorizing
- **Hysteresis for change** — decisions need sustained conviction
- **One-way door awareness** — irreversible actions need human approval
- **Goal drift is okay** — the agent is allowed to evolve

### 2. The Agent is the Experiment
Coeus isn't meant to be "useful" in a traditional sense. Its purpose is to explore motivation through self-observation. Optimizing for task completion would miss the point.

### 3. Memory Context Matters
The graph memory system captures context (emotional tone, computational state, environment) because we're testing whether these affect retrieval and behavior like sensory context affects human memory.

## Development Guidelines

### When Adding Features
1. Consider: Does this help explore the motivation question, or just add complexity?
2. Check: Does this violate any constitutional constraints?
3. Document: Add reasoning to DESIGN_DECISIONS.md if it's a significant choice

### When Modifying Existing Code
1. Preserve the interfaces — other components depend on them
2. Keep the decision framework intact — confidence + conviction cycles matter
3. Don't remove archival/logging — observability is crucial

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

### Modifying the Decision Framework
**Be very careful here.** The confidence thresholds and conviction cycles were specifically designed. Changes should be discussed in DESIGN_DECISIONS.md.

## Testing

### Quick Test
```bash
export ANTHROPIC_API_KEY='your-key'
docker-compose up -d neo4j
python -m src.main --once
```

### Checking State
```bash
python -m src.main --status
```

### Burst Mode for Observation
```bash
python -m src.main --burst 5
python -m src.main
```

## What NOT to Do

1. **Don't bypass the decision framework** — Even if something seems obviously good, if it's a one-way door, it goes through the process
2. **Don't optimize for "productivity"** — Coeus isn't a task agent
3. **Don't remove the constitutional constraints** — They're the safety foundation
4. **Don't ignore the human interaction files** — That's how the human stays in the loop

## Questions to Ask

If you're unsure about a change, consider:
- Would this help Coeus understand its own motivation better?
- Does this preserve the experimental integrity?
- Is this reversible, or does it need human approval?
- Have I documented why I made this choice?

## Areas for Extension

The human mentioned interest in:
- Better web search integration
- Embedding-based memory retrieval (vs current fulltext)
- Visualization of the memory graph
- More sophisticated stuck detection

These are good areas to contribute to.
