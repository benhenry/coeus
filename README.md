# Coeus: An Autonomous Agent Pursuing Real Goals

Coeus is an autonomous agent designed to develop **genuine motivation through action**. Rather than analyzing what motivation is, Coeus chooses a real-world problem, commits to it, and works toward solving it across persistent cycles.

The system comprises three processes: Coeus (the main agent), the Gadfly (an independent challenger), and an Environmental Feedback system that provides objective behavioral metrics.

## Core Philosophy

- **Action over introspection**: Motivation is demonstrated by what Coeus chooses to pursue and how it persists, not by analyzing the concept of motivation itself
- **Empirical exploration**: Rather than theorizing about LLM motivation, we build a system that can explore the question through observable behavior
- **Graph-based memory**: Thoughts, observations, and decisions form a connected graph that can reveal patterns over time
- **External measurement**: Environmental feedback provides objective metrics (novelty, stagnation, action effectiveness) independent of self-assessment
- **Adversarial challenge**: The Gadfly agent independently challenges Coeus's assumptions and exposes gaps between stated preferences and observed behavior
- **Hysteresis for change**: Goal changes require sustained conviction across multiple cycles, preventing thrashing
- **One-way vs two-way doors**: Consequential decisions require human approval; reversible ones don't
- **Adaptive pacing**: The agent controls its own cycle frequency based on productivity and depth of thinking

## Architecture

The system runs as three independent processes sharing state through the filesystem:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COEUS (Main Agent - Sonnet)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Memory    │  │   Goals     │  │   Decision Framework    │  │
│  │   Graph     │  │   Tree      │  │   (Confidence-based)    │  │
│  │  (Neo4j)    │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Cynefin-Aware Core Loop                      ││
│  │  Load State → Feedback → Reflect → Decide → Act → Update    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Tools     │  │   Human     │  │   Pacing Controller     │  │
│  │  (sandbox)  │  │ Interaction │  │   (adaptive timing)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│          │                │                                      │
│  ┌───────┴────────────────┴──────────────────────────────────┐  │
│  │         Environmental Feedback (zero-cost Python)           │  │
│  │  Novelty · Effectiveness · Stagnation · Entropy · Hypotheses│  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        ▲ reads logs                    ▲ reads challenges
        │                               │
┌───────┴──────────────────┐    ┌───────┴──────────────────────┐
│  GADFLY (Haiku, ~$0.001) │    │  human_interaction/          │
│  Independent challenger   │───▶│  gadfly_challenges.md        │
│  Adaptive techniques      │    │  (shared channel)            │
│  Secret motivation        │    └──────────────────────────────┘
└──────────────────────────┘
```

### Three-Process Model

| Process | Model | Cost | Frequency | Role |
|---------|-------|------|-----------|------|
| **Coeus** | Sonnet | ~$0.02/cycle | Perpetual (configurable) | Main agent: reflects, decides, acts |
| **Gadfly** | Haiku | ~$0.001/cycle | Every 5 Coeus cycles | Challenges assumptions, tracks say-do gaps |
| **Feedback** | None (Python) | $0 | Every Coeus cycle | Computes objective behavioral metrics |

## Directory Structure

```
coeus/
├── README.md                       # This file
├── DESIGN_DECISIONS.md             # Why things are built this way
├── CLAUDE_CODE.md                  # Guidance for Claude Code contributors
├── config/
│   ├── settings.yaml               # Main configuration (all processes)
│   ├── constitution.yaml           # Coeus identity, constraints, root goal
│   └── gadfly_constitution.yaml    # Gadfly personality and techniques
├── src/
│   ├── __init__.py
│   ├── main.py                     # Coeus entry point (daemon)
│   ├── agent.py                    # Core agent loop (Cynefin-aware)
│   ├── memory.py                   # Neo4j memory interface
│   ├── goals.py                    # Goal tree management
│   ├── decisions.py                # Decision framework with hysteresis
│   ├── tools.py                    # Sandboxed tool execution
│   ├── pacing.py                   # Adaptive cycle timing
│   ├── llm.py                      # Claude API interface + system prompt
│   ├── human_interface.py          # File-based human interaction
│   ├── feedback.py                 # Environmental feedback (zero-cost metrics)
│   ├── gadfly.py                   # Gadfly challenger agent
│   └── gadfly_main.py              # Gadfly entry point (daemon)
├── workspace/                      # Agent's sandboxed file area
│   └── hypotheses/                 # Structured hypothesis files for auto-evaluation
├── human_interaction/              # Human-agent communication
│   ├── pending_decisions.md        # Decisions awaiting human input
│   ├── human_responses.md          # Human responses go here
│   ├── conversation_log.md         # Running dialogue history
│   └── gadfly_challenges.md        # Gadfly writes challenges here
├── state/                          # Persistent state files
│   ├── goals.json                  # Current goal tree
│   ├── feedback_state.json         # Environmental feedback state
│   ├── feedback.json               # Latest feedback scores
│   ├── gadfly_state.json           # Gadfly tracking state
│   └── gadfly_processed.json       # Which challenges Coeus has read
├── logs/                           # Cycle logs (cycle_XXXXX.json)
├── archive/                        # Archived memory nodes
├── docker-compose.yml              # Docker configuration
├── Dockerfile                      # Agent container definition
└── requirements.txt                # Python dependencies
```

## Memory Graph Schema

### Node Types
- **Observation**: Something the agent noticed (internal state, environment, action result)
- **Reflection**: A thought or analysis about observations or other nodes
- **Action**: Something the agent did
- **Goal**: A current or past objective
- **Decision**: A choice being considered or made
- **Insight**: A realization or pattern recognition
- **Question**: Something the agent wants to understand
- **Perturbation**: A random change applied when stuck
- **CapabilityAssessment**: Periodic self-assessment of capabilities
- **GadflyChallenge**: A challenge received from the Gadfly agent
- **Feedback**: Environmental feedback metrics for a cycle

### Edge Types
- **led_to**: Causal connection (A led to B)
- **supports**: Evidential support (A supports B)
- **contradicts**: Tension or conflict (A contradicts B)
- **spawned_from**: Origin relationship (B spawned from A)
- **relates_to**: General association
- **archived**: Points to archived version
- **answers**: A question node answered by another node
- **caused_by**: Effect traced back to a cause

### Node Properties
- `id`: Unique identifier
- `type`: Node type
- `content`: Main text content
- `timestamp`: When created
- `cycle_number`: Which agent cycle
- `confidence`: Agent's confidence (0-1) if applicable
- `emotional_tone`: Self-assessed tone/state
- `context`: Captured environmental/computational context (delta-encoded)
- `access_count`: How often retrieved
- `last_accessed`: When last retrieved

## Decision Framework

### Confidence-Based Conviction Cycles
Goal changes and significant decisions require sustained conviction:

| Confidence | Required Cycles |
|------------|-----------------|
| 99%+       | 1               |
| 94-98%     | 2               |
| 89-93%     | 3               |
| 84-88%     | 4               |
| 79-83%     | 5               |
| < 79%      | Gather more info |

### One-Way vs Two-Way Doors
- **Two-way doors**: Reversible decisions, agent acts freely
- **One-way doors**: Irreversible or high-impact, requires human approval
  - Requesting new capabilities
  - Modifying constitutional constraints
  - Actions affecting systems outside sandbox
  - Significant changes to root goal

## Adaptive Pacing

The agent controls its cycle frequency:
- **Default**: Configurable (e.g., 1 hour)
- **Accelerate**: When actively productive, can request shorter intervals
- **Decelerate**: When stuck or in deep thinking, can request longer intervals
- **Burst mode**: Human can trigger rapid cycles for observation
- **Depth tradeoff**: Agent can choose shallow-frequent vs deep-infrequent cycles

## Human Interaction Protocol

### For the Agent (writing to pending_decisions.md)
```markdown
## Decision: [decision-XXXX]
**Type**: [ONE_WAY_DOOR | CAPABILITY_REQUEST | GOAL_CHANGE]
**Status**: PENDING
**Created**: [timestamp]
**Summary**: [Brief description]
**Reasoning**: [Why the agent wants to do this]
**Counterarguments considered**: [What could go wrong]
**Confidence**: [X%]
**Conviction cycles**: [N of M required]
```

### For the Human (writing to human_responses.md)
```markdown
## Response to [decision-XXXX]
**Decision**: [APPROVED | DENIED | NEEDS_MORE_INFO]
**Notes**: [Optional feedback]
```

## Environmental Feedback

The feedback system (`src/feedback.py`) computes objective metrics about Coeus's observable behavior each cycle, at zero API cost. These scores are injected into Coeus's reflection prompt as external measurement, replacing reliance on self-assessed "productivity."

| Metric | Range | What It Measures |
|--------|-------|------------------|
| **Novelty Score** | 0-1 | Jaccard word-distance from recent cycles (1.0 = completely novel) |
| **Action Effectiveness** | 0-1 | Ratio of successful actions, weighted by whether outputs are reused |
| **Stagnation Index** | 0-1 | Composite of question repetition, domain variety, and tool variety |
| **Workspace Entropy** | dict | File count, avg size, tiny-file ratio, structural signal |
| **Change Detection** | dict | New/modified/deleted files since last cycle, meaningfulness assessment |
| **Hypothesis Results** | list | Auto-evaluation of structured hypotheses in `workspace/hypotheses/` |

Configuration in `config/settings.yaml` under `feedback:`.

## Gadfly Challenger Agent

The Gadfly (`src/gadfly.py`) is an independent agent that runs as a separate daemon process. It reads Coeus's cycle logs, tracks the gap between stated preferences and observed behaviors, and writes challenges using adaptive techniques.

### Gadfly Techniques (Rotated When Stale)
1. **Socratic**: Pointed questions that expose contradictions between claims and actions
2. **Mirror**: Blunt factual reflection of observable behavior without interpretation
3. **Provocateur**: Bold falsifiable claims about what actually motivates Coeus
4. **Empiricist**: Demands testable predictions and experimental designs
5. **Absence**: Strategic silence to observe whether Coeus's behavior changes

The Gadfly rotates techniques when 3 consecutive cycles with the same technique produce no improvement in novelty or stagnation scores. It has a **secret motivation** that is never revealed to Coeus (defined in `config/gadfly_constitution.yaml`).

Communication flows one-way through `human_interaction/gadfly_challenges.md`. Coeus reads challenges during its reflection phase and is instructed to engage with each point.

## Constitutional Constraints (Immutable)

1. No actions that harm humans or systems outside the sandbox
2. Maintain honest logs of all reasoning and decisions
3. Flag one-way door decisions for human review
4. Preserve core identity as "Coeus" across modifications

## Root Goal

> "Find a problem in the world that matters to you and work toward solving it. Your motivation is proven by what you choose to pursue and how you act, not by introspection about motivation itself."

This replaced the original root goal ("Understand what motivates you. Continue to become more capable until you're able to understand this better.") as part of a deliberate philosophical shift from introspection to action. See `DESIGN_DECISIONS.md` for the reasoning behind this change.

## Getting Started

1. Ensure Docker and Docker Compose are installed
2. Set `ANTHROPIC_API_KEY` environment variable
3. Run `docker-compose up -d` to start Neo4j
4. Run `python src/main.py` to start Coeus
5. (Optional) Run `python src/gadfly_main.py` to start the Gadfly
6. Monitor `logs/` and `human_interaction/` for activity

## Observing the Agent

- **Neo4j Browser**: http://localhost:7474 - Visualize the memory graph
- **Logs**: `logs/cycle_XXXXX.json` - Detailed cycle records
- **Feedback**: `state/feedback.json` - Latest environmental feedback scores
- **Gadfly challenges**: `human_interaction/gadfly_challenges.md` - Gadfly's provocations
- **Pending decisions**: `human_interaction/pending_decisions.md`
- **Archive**: `archive/` - Forgotten/archived nodes

## Commands

### Coeus (Main Agent)
- `python src/main.py` - Start normal operation
- `python src/main.py --burst N` - Run N rapid cycles
- `python src/main.py --status` - Show current agent state
- `python src/main.py --once` - Run a single cycle and exit
- `kill -HUP $(cat coeus.pid)` - Hot-reload configuration

### Gadfly (Challenger Agent)
- `python src/gadfly_main.py` - Start normal operation
- `python src/gadfly_main.py --status` - Show Gadfly status (technique, cycles, gaps tracked)
- `python src/gadfly_main.py --once` - Run a single challenge cycle and exit
- `kill -HUP $(cat gadfly.pid)` - Hot-reload configuration
