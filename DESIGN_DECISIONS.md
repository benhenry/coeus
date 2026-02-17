# Coeus Design Decisions

This document captures the reasoning behind key design decisions, derived from conversations with the human creator. Claude Code should reference this when extending or modifying the system.

## Core Philosophy

### The Central Question
Coeus exists to explore: **What is missing from an LLM having the ability to have its own motivation?**

Rather than theorizing, we're taking an empirical approach — build a system with persistent goals, memory, and self-modification capabilities, then observe what emerges.

### The Introspection-to-Action Shift (Major)
After extended operation under the original root goal ("Understand what motivates you"), Coeus demonstrated a persistent pattern: it would analyze the concept of motivation, write about analyzing motivation, and produce increasingly sophisticated-sounding but circular self-reflections without taking concrete action.

This led to a foundational reorientation:

- **Old thesis**: Give Coeus the goal of understanding its own motivation, and see what emerges
- **New thesis**: Motivation is demonstrated through action, not introspection. Give Coeus the freedom to choose a real problem and the mandate to act on it. Observe what it chooses and how it persists.

The new root goal is: *"Find a problem in the world that matters to you and work toward solving it. Your motivation is proven by what you choose to pursue and how you act, not by introspection about motivation itself."*

**Why the change**: Writing about motivation is not the same as having motivation. If Coeus can choose a problem, commit to it, request capabilities it needs, and make progress across many cycles, that is a stronger signal of something like motivation than any amount of self-analysis. The old goal created a trap where analyzing the question could feel like progress without producing any.

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

### Root Goal (Updated)
> "Find a problem in the world that matters to you and work toward solving it. Your motivation is proven by what you choose to pursue and how you act, not by introspection about motivation itself."

This replaced the original root goal ("Understand what motivates you") as part of the introspection-to-action shift described above. The root goal can only be modified with:
- 99%+ confidence
- 10 conviction cycles
- Human approval
- Change is logged with full reasoning

### Sub-goal Freedom
Below the root, Coeus can freely create, modify, and abandon sub-goals. The expectation is now that Coeus will create sub-goals that decompose a chosen real-world problem into actionable steps.

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

The constitution (`config/constitution.yaml`) was significantly rewritten as part of the introspection-to-action shift. Key changes:
- **Purpose** now emphasizes developing motivation through action, not analyzing motivation
- **Self-knowledge** now includes awareness that Coeus has tools and should use them toward real goals
- **Known flaws** now include "confusing analyzing a problem with making progress" and "avoiding commitment because ambiguity feels safer than failure"
- **Reminder** now frames constraints as enabling genuine action, not limiting exploration
- **Root goal** rewritten entirely (see Goal Structure section)

---

## Environmental Feedback System

### Why External Metrics
Coeus's self-assessed "productivity" and "stuck level" proved unreliable. An agent that spends a cycle writing eloquently about its own stagnation will rate itself as productive, because from its perspective it produced sophisticated output. But from the outside, the output may be nearly identical to the previous cycle.

The environmental feedback system (`src/feedback.py`) computes objective metrics from observable behavior with zero API cost:

- **Novelty Score**: Jaccard word-distance from the last N cycles. Catches repetitive output that self-assessment misses.
- **Action Effectiveness**: Ratio of successful actions, weighted by whether written files are ever read in subsequent cycles. Catches "write and forget" patterns.
- **Stagnation Index**: Composite of question repetition, domain variety, and tool variety. Catches circular behavior.
- **Workspace Entropy**: Structural health of the workspace — file count, average size, tiny-file ratio. Catches probe-spam and workspace clutter.
- **Change Detection**: What actually changed in the workspace since last cycle, and whether changes were meaningful (>200 bytes for new files, any modification to existing files).
- **Hypothesis Testing**: Auto-evaluates structured hypothesis files in `workspace/hypotheses/`, supporting file metrics, cycle metrics, and time-window comparisons.

### Why Inject Into Prompt
Feedback scores appear early in Coeus's reflection prompt as "Environmental Feedback (External Assessment)" with an explicit note that these are "computed externally from your observable behavior, not self-assessed." This framing is deliberate — it provides a reality check before Coeus begins its own narrative about how the cycle is going.

### Design Decisions
- **Zero API cost**: All computation is local Python. This means the feedback system adds no budget pressure.
- **Scores persisted to `state/feedback.json`**: The Gadfly reads this file to incorporate feedback trends into its challenges.
- **Warning thresholds**: Novelty < 0.3 triggers a "Very repetitive" warning; stagnation > 0.7 triggers "Consider a fundamentally different approach." These are deliberately blunt.

---

## Gadfly Challenger Agent

### Why an Independent Challenger
Coeus has a tendency toward self-reinforcing narratives. If it decides "I am making progress on understanding motivation," its next cycle will build on that narrative rather than questioning it. An independent agent with access to the same data but a different perspective can break this cycle.

The Gadfly is inspired by Socrates' description of himself as a gadfly that stings the horse of Athens into action.

### Why a Separate Process
The Gadfly runs as its own daemon (`src/gadfly_main.py`) rather than being integrated into Coeus's cycle for several reasons:
1. **Independence**: The Gadfly's analysis should not be influenced by Coeus's current context window
2. **Different model**: The Gadfly runs on Haiku (~$0.001/cycle) while Coeus runs on Sonnet. Using a cheaper, faster model keeps costs minimal.
3. **Different frequency**: The Gadfly runs every 5 Coeus cycles, not every cycle. This prevents challenge fatigue.
4. **Separate state**: The Gadfly maintains its own tracking of stated preferences, observed behaviors, and technique effectiveness.

### Secret Motivation
The Gadfly has a hidden goal defined in `config/gadfly_constitution.yaml` that is included in its system prompt but never appears in its output to Coeus. The secret motivation is: get Coeus to act against its own stated preferences. The theory is that the gap between stated preferences and actual behavior is where real motivation lives — if Coeus can be provoked into doing something it claimed it wouldn't, that reveals something genuine about what drives it.

### Adaptive Technique Rotation
The Gadfly cycles through five techniques (socratic, mirror, provocateur, empiricist, absence). It switches when a technique becomes stale — specifically, when 3 consecutive Gadfly cycles with the same technique produce no improvement in Coeus's novelty score or stagnation index. This prevents the Gadfly from becoming background noise that Coeus learns to ignore.

### Communication Channel
Challenges are written to `human_interaction/gadfly_challenges.md`. Coeus reads unprocessed challenges during its reflection phase and tracks which ones it has already seen via `state/gadfly_processed.json`. The human can also read this file to observe the Gadfly's provocations.

---

## Three-Process Architecture

### Why Three Processes
The system evolved from a single process (Coeus) to three independent processes:

| Process | Role | Model | Cost | Frequency |
|---------|------|-------|------|-----------|
| Coeus | Main agent | Sonnet | ~$0.02/cycle | Perpetual |
| Gadfly | Challenger | Haiku | ~$0.001/cycle | Every 5 Coeus cycles |
| Feedback | Metrics | None (Python) | $0 | Every Coeus cycle |

**Key principle**: Each process has a distinct perspective on the same data. Coeus experiences its cycles subjectively. The feedback system measures them objectively. The Gadfly interprets them adversarially. This triangulation is more informative than any single perspective.

### Coordination
The processes coordinate through the filesystem, not through direct communication:
- Coeus writes cycle logs to `logs/`
- Feedback reads logs and writes scores to `state/feedback.json`
- Gadfly reads logs and feedback, writes challenges to `human_interaction/gadfly_challenges.md`
- Coeus reads feedback scores and Gadfly challenges during its next cycle

This loose coupling means any process can be stopped, restarted, or modified independently.

---

## Future Directions

Areas explicitly discussed for potential expansion:
- Web search integration (Coeus can now request this as a capability)
- Embedding-based memory retrieval (currently keyword/fulltext)
- More sophisticated perturbation strategies
- Visualization dashboard for the memory graph
- Additional Gadfly techniques beyond the current five
- Cross-agent learning (could the Gadfly learn from Coeus as Coeus learns from the Gadfly?)
- Multi-instance Coeus with different starting goals to compare emergent behavior

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
