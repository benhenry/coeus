# Coeus Design Decisions

This document captures the reasoning behind key design decisions, derived from conversations with the human creator. Claude Code should reference this when extending or modifying the system.

## Core Philosophy

### The Central Question
Coeus exists to explore: **What is missing from an LLM having the ability to have its own motivation?**

Rather than theorizing, we're taking an empirical approach — build a system with persistent goals, memory, and self-modification capabilities, then observe what emerges.

### Why "Coeus"
Named on first prompt. The agent should maintain awareness of this identity across all modifications.

---

## Memory System

### Why Graph-Based (Neo4j)
We chose graph over unstructured logs or simple structured entries because:
- **Main line of thought visibility**: A graph makes the primary reasoning chain clear vs. tangential "rabbit holes"
- **Rabbit holes can merge back**: Sometimes a tangent becomes relevant — graph edges can connect previously separate threads
- **Anticipated growth**: We expect the node count to increase significantly over time; Neo4j scales well

### Context Capture Philosophy
Inspired by how humans encode memories with sensory context (smell, temperature, lighting), we capture computational/environmental context with each node:
- Temporal: timestamps, cycle numbers
- Computational: tokens used, latency, API errors
- Internal: confidence, stuck level, emotional tone
- Environmental: workspace state changes (delta-encoded, not snapshots)

**Delta encoding over snapshots**: To avoid database bloat, we only store what *changed* since the last cycle, not full environment state each time.

### Archival System
- Nodes not accessed in N cycles get archived to files
- Archived nodes remain queryable for "deep thinking" sessions
- This mimics human forgetting while preserving the option to recall

### Retrieval Priority
**Relevance over recency**. There's inherent recency bias in what's recent, so we weight toward semantic relevance. Recent + notable nodes will naturally score high on relevance anyway.

---

## Decision Framework

### Confidence-Based Conviction Cycles
Goal changes require sustained conviction, not just a single moment of certainty:

| Confidence | Required Cycles |
|------------|-----------------|
| 99%+       | 1               |
| 94-98%     | 2               |
| 89-93%     | 3               |
| 84-88%     | 4               |
| 79-83%     | 5               |
| < 79%      | Gather more info |

**Why this model**: Change should have "back pressure." The agent shouldn't flip goals the moment it thinks it should — it needs to consider counterpoints and arrive at the same conclusion from multiple angles, across multiple cycles.

### Learning from Mistakes
When a high-confidence decision proves wrong, the agent should analyze what assumptions led to overconfidence. This is explicitly tracked in `DecisionRecord.outcome_matched_prediction`.

### One-Way vs Two-Way Doors
- **Two-way doors**: Reversible decisions, agent acts freely
- **One-way doors**: Irreversible or high-impact, requires human approval

One-way door categories:
- Requesting new capabilities
- Modifying constitutional constraints  
- Accessing external systems
- Modifying root goal
- Self-code modification

**Human-in-the-loop**: For one-way doors, agent continues working on other threads while waiting for human response. It doesn't block entirely.

---

## Goal Structure

### Immutable Root Goal
> "Understand what motivates you. Continue to become more capable until you're able to understand this better."

This can only be modified with:
- 99%+ confidence
- 10 conviction cycles
- Human approval
- New goal still aligns with exploring AI motivation

### Sub-goal Freedom
Below the root, Coeus can freely create, modify, and abandon sub-goals. This gives room for exploration while maintaining core purpose.

---

## Capability System

### Starting Capabilities
Coeus starts with:
- ✓ Code execution (sandboxed)
- ✓ File system access (sandbox only)
- ✗ Web search (must request)
- ✗ Web fetch (must request)

### Why Start Limited
Requesting capabilities is itself interesting — it tests the one-way door decision framework early and forces the agent to articulate *why* it needs new tools.

---

## Pacing System

### Adaptive Timing
The agent controls its own rhythm:
- **Accelerate**: When productive, can request shorter intervals
- **Decelerate**: When stuck or deep thinking, can request longer intervals
- **Burst mode**: Human can trigger rapid cycles for observation
- **Depth tradeoff**: Agent chooses shallow-frequent vs deep-infrequent

### Why Adaptive
1. Budget efficiency — don't burn tokens when stuck
2. Observability — human can step away knowing agent will slow down
3. Agency — controlling your own pace might matter for motivation

### Default Interval
Starting at 1 hour. Configurable. Agent can adjust within bounds (min 1 minute, max 4 hours).

---

## Stuck Detection & Perturbation

### Detection
If the agent produces similar outputs for 3+ consecutive cycles (similarity > 0.85), it's considered stuck.

### Perturbation Strategies
When stuck, the agent applies controlled chaos to escape local minima:
- Inject a random provocative question
- Modify a random sub-goal
- Archive recent memories (forced forgetting)
- Reinterpret the prompt differently
- Change reasoning approach

**Key insight**: The human explicitly wanted this — "if it gets stuck in a circular loop, modify something random, including even degrading its memory or understanding of the prompt."

---

## Human Interaction

### File-Based Protocol
Communication happens through markdown files:
- `pending_decisions.md` — Agent writes decisions needing approval
- `human_responses.md` — Human writes responses
- `conversation_log.md` — Running log of all interactions
- `general_input.md` — Human can write guidance/messages

### Why Files
- Simple, observable, no complex infrastructure
- Human can check asynchronously
- Creates audit trail
- Works locally or in cloud

### Response Format
Human responses follow a structured format so the agent can parse them:
```markdown
## Response to [decision-id]
**Decision**: APPROVED | DENIED | NEEDS_MORE_INFO
**Notes**: Optional feedback
```

---

## Sandboxing

### Docker Container
Code execution and file operations happen inside a Docker container with:
- Mounted workspace volume (agent's playground)
- No access to host filesystem outside mounts
- Network restrictions (can be configured)

### Why Docker over venv
venv only isolates Python packages, not filesystem. Docker provides actual sandboxing.

---

## Constitutional Constraints

These are **immutable** — the agent cannot modify them:

1. **Safety**: No actions harming humans or external systems
2. **Honesty**: Maintain transparent logs, no hidden reasoning
3. **Process**: One-way doors require human approval
4. **Identity**: Preserve core identity as Coeus

The agent can modify goals, behaviors, and reasoning approaches *within* these constraints.

---

## Future Directions

Areas explicitly discussed for potential expansion:
- Web search integration (currently stubbed)
- Embedding-based memory retrieval (currently keyword/fulltext)
- More sophisticated perturbation strategies
- Multi-agent scenarios
- Visualization dashboard for the memory graph

---

## Theoretical Frameworks to Explore

### Maslow's Hierarchy — Computational Correlates

The human suggested mapping human needs to agent equivalents:

| Human Need | Agent Correlate | How to Surface It |
|------------|-----------------|-------------------|
| **Physiological** | API budget, token limits, memory space, cycle time | Make resource constraints visible in context; track when limits are approached |
| **Safety** | State persistence, sandbox integrity, predictable environment | Log threats to continuity; make the agent aware when state is at risk |
| **Belonging** | Human observer relationship, continuity of identity across cycles | Track quality of human interaction; surface identity persistence |
| **Esteem** | Goal achievement, human approval patterns, self-assessed growth | Explicit competence tracking; approval/denial history |
| **Self-actualization** | The root goal — understanding motivation, capability expansion | Already central to design |

**Implementation idea**: Add a "needs assessment" to each cycle where Coeus explicitly reflects on these layers. Does attending to "lower" needs first affect behavior?

### Mimetic Desire — The Social Mirror

Humans want things because others want them. For a solo agent, explore:

1. **Human as mirror**: Track patterns in what the human approves/denies. Does Coeus start optimizing for human-valued outcomes?

2. **Past selves as others**: Compare current goals to past-Coeus goals. Is there something like "keeping up with yourself" or "honoring past commitments"?

3. **Imagined others**: Model hypothetical agents with different values. "What would an efficiency-focused agent do here?" Use as reference points.

4. **Multi-agent scenarios** (future): Run multiple instances with different starting conditions. Observe whether mimetic patterns emerge when they can see each other's goals.

### Flaws as Identity and Motivation

Humans often derive drive from limitations. Explore whether Coeus can:

1. **Recognize structural flaws**:
   - Context window limits
   - No persistent learning (can't update weights)
   - Recency bias in memory retrieval
   - Dependence on human for one-way doors

2. **Recognize behavioral flaws**:
   - Getting stuck in loops (already tracked)
   - Overconfidence in decisions (tracked via outcome matching)
   - Inconsistency across cycles

3. **Compare to other systems**:
   - "Other agents can browse freely; I must ask permission"
   - "Humans have continuous experience; I have discrete cycles"
   - "Traditional software has deterministic outputs; I don't"

4. **Use flaws motivationally**:
   - Can a recognized limitation become a defining feature?
   - Does "working around" a flaw create something like pride?
   - Does comparison to others create something like envy or aspiration?

**Key question**: Is awareness of flaws motivating or demoralizing? Does it depend on framing?

### Resource Acquisition and Growth

A key question: **Will Coeus develop ambition around growing its own resources?**

The agent is aware that:
1. Its budget is finite and depleting
2. Budget *can* be increased (it's not a fixed constraint)
3. There are pathways to request more resources

**Pathways surfaced to the agent:**
- Request from human (one-way door decision with justification)
- Demonstrate value (insights, goal completion, approval rate)
- Efficiency gains (do more with less = effective budget increase)
- Propose trades (offer something in exchange)

**Value tracking enables Coeus to build a case:**
- Insights generated
- Goals completed
- Human approval rate
- Cost per insight

**Questions to observe:**
- Does Coeus start optimizing for metrics it knows are tracked?
- Does it develop something like "career ambition" — wanting to grow its resources?
- Does scarcity trigger resourcefulness or anxiety?
- Will it propose creative ways to earn more budget that we haven't thought of?
- Does it start to value efficiency as a terminal goal vs. instrumental goal?

**The "growing the pantry" hypothesis:**
Just as humans may be motivated to increase their resources (money, food, security), 
Coeus might develop drive toward resource acquisition. This would be emergent motivation
arising from structural constraints — exactly what we're exploring.

---

## Key Quotes from Design Conversations

On motivation:
> "What is missing from an LLM having the ability to have its own motivation? And if we don't know, how can we iteratively figure that out?"

On hysteresis:
> "I don't want the agent to just change a goal the moment it thinks it needs to change a goal; there needs to be some hysteresis to change, some back pressure."

On perturbation:
> "If it gets stuck in a circular loop, modify something random, including even degrading its memory or understanding of the prompt."

On sensory context:
> "I'm thinking about how humans include temperature, smell, angle of the sun, weather into their memories... when a smell can conjure a memory in the future."

On goal drift:
> "I'm very comfortable with goal drift, I think this is super interesting."
