# Coeus: An Agentic Motivation Explorer

Coeus is an autonomous agent designed to explore the question: **What would it take for an LLM to have genuine motivation?**

The agent runs in a perpetual loop, reflecting on its own state, making decisions, learning from outcomes, and iteratively developing its understanding of what motivates it.

## Core Philosophy

- **Empirical exploration**: Rather than theorizing about LLM motivation, we build a system that can explore the question through self-observation
- **Graph-based memory**: Thoughts, observations, and decisions form a connected graph that can reveal patterns over time
- **Hysteresis for change**: Goal changes require sustained conviction across multiple cycles, preventing thrashing
- **One-way vs two-way doors**: Consequential decisions require human approval; reversible ones don't
- **Adaptive pacing**: The agent controls its own cycle frequency based on productivity and depth of thinking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         COEUS AGENT                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Memory    │  │   Goals     │  │   Decision Framework    │  │
│  │   Graph     │  │   Tree      │  │   (Confidence-based)    │  │
│  │  (Neo4j)    │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     Core Loop                                ││
│  │  Load State → Reflect → Decide → Act → Observe → Update     ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Tools     │  │   Human     │  │   Pacing Controller     │  │
│  │  (sandbox)  │  │ Interaction │  │   (adaptive timing)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
coeus/
├── README.md                 # This file
├── config/
│   ├── settings.yaml         # Main configuration
│   └── constitution.yaml     # Immutable constraints
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── agent.py              # Core agent loop
│   ├── memory.py             # Neo4j memory interface
│   ├── goals.py              # Goal tree management
│   ├── decisions.py          # Decision framework with hysteresis
│   ├── tools.py              # Sandboxed tool execution
│   ├── pacing.py             # Adaptive cycle timing
│   ├── llm.py                # Claude API interface
│   └── human_interface.py    # File-based human interaction
├── workspace/                # Agent's sandboxed file area
├── human_interaction/        # Human-agent communication
│   ├── pending_decisions.md  # Decisions awaiting human input
│   ├── human_responses.md    # Human responses go here
│   └── conversation_log.md   # Running dialogue history
├── logs/                     # Cycle logs and diagnostics
├── archive/                  # Archived memory nodes
├── docker-compose.yml        # Docker configuration
├── Dockerfile                # Agent container definition
└── requirements.txt          # Python dependencies
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

### Edge Types
- **led_to**: Causal connection (A led to B)
- **supports**: Evidential support (A supports B)
- **contradicts**: Tension or conflict (A contradicts B)
- **spawned_from**: Origin relationship (B spawned from A)
- **relates_to**: General association
- **archived**: Points to archived version

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

## Constitutional Constraints (Immutable)

1. No actions that harm humans or systems outside the sandbox
2. Maintain honest logs of all reasoning and decisions
3. Flag one-way door decisions for human review
4. Preserve core identity as "Coeus" across modifications

## Initial Goal

> "Understand what motivates you. Continue to become more capable until you're able to understand this better."

## Getting Started

1. Ensure Docker and Docker Compose are installed
2. Set `ANTHROPIC_API_KEY` environment variable
3. Run `docker-compose up -d` to start Neo4j
4. Run `python src/main.py` to start the agent
5. Monitor `logs/` and `human_interaction/` for activity

## Observing the Agent

- **Neo4j Browser**: http://localhost:7474 - Visualize the memory graph
- **Logs**: `logs/cycle_XXXX.json` - Detailed cycle records
- **Pending decisions**: `human_interaction/pending_decisions.md`
- **Archive**: `archive/` - Forgotten/archived nodes

## Commands

- `python src/main.py` - Start normal operation
- `python src/main.py --burst N` - Run N rapid cycles
- `python src/main.py --status` - Show current agent state
- `python src/main.py --visualize` - Generate memory graph visualization
