"""
Core Agent Loop for Coeus

This is the heart of the agent - the perpetual loop that:
1. Loads state and context
2. Reflects on current situation
3. Makes decisions
4. Takes actions
5. Observes results
6. Updates memory and goals
7. Adjusts pacing and sleeps
"""

import json
import random
import re
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from llm import LLMInterface, LLMResponse, build_system_prompt
from memory import (
    MemoryGraph, MemoryNode, NodeType, EdgeType,
    create_memory_node, ContextCapture
)
from goals import GoalTree, format_goals_for_prompt
from decisions import (
    DecisionFramework, DecisionRecord, DecisionType, DecisionStatus,
    format_decision_for_prompt
)
from pacing import PacingController, CycleMetrics, calculate_output_similarity
from tools import (
    SandboxedTools, CapabilityManager, WebSearchTool,
    format_tool_result, get_available_tools_description
)
from human_interface import HumanInterface, format_human_input_for_prompt


@dataclass
class CycleState:
    """State captured during a single cycle."""
    cycle_number: int
    start_time: str
    
    # LLM interactions
    reflection_response: Optional[LLMResponse] = None
    action_response: Optional[LLMResponse] = None
    
    # Parsed outputs
    observations: list[str] = None
    reflections: list[str] = None
    intended_actions: list[dict] = None
    questions: list[str] = None
    meta_observations: list[str] = None
    
    # Execution results
    action_results: list[dict] = None
    
    # Self-assessments
    productivity: float = 0.5
    stuck_level: float = 0.0
    emotional_tone: str = ""
    confidence: float = 0.5
    
    # Pacing preferences
    wants_faster: bool = False
    wants_slower: bool = False
    requested_depth: str = "normal"
    
    def __post_init__(self):
        self.observations = self.observations or []
        self.reflections = self.reflections or []
        self.intended_actions = self.intended_actions or []
        self.questions = self.questions or []
        self.meta_observations = self.meta_observations or []
        self.action_results = self.action_results or []


class CoeusAgent:
    """
    The main Coeus agent.
    
    Orchestrates the perpetual loop of reflection, decision, action, and learning.
    """
    
    def __init__(self, config: dict, base_path: str = "."):
        self.config = config
        self.base_path = Path(base_path)
        
        # Initialize components
        self.llm = LLMInterface(
            model=config['llm']['model'],
            max_tokens=config['llm']['max_tokens']
        )
        
        # Memory graph
        neo4j_config = config['neo4j']
        self.memory = MemoryGraph(
            uri=neo4j_config['uri'],
            user=neo4j_config['user'],
            password=neo4j_config['password'],
            database=neo4j_config['database']
        )
        
        # Goals
        self.goals = GoalTree(
            root_goal_content=config.get('root_goal', 
                "Understand what motivates you. Continue to become more capable until you're able to understand this better."),
            state_path=str(self.base_path / "state" / "goals.json")
        )
        
        # Decisions
        self.decisions = DecisionFramework(
            human_interaction_path=str(self.base_path / "human_interaction")
        )
        self.decisions.load_state(str(self.base_path / "state" / "decisions.json"))
        
        # Pacing
        self.pacing = PacingController(
            default_interval=config['pacing']['default_interval_seconds'],
            min_interval=config['pacing']['min_interval_seconds'],
            max_interval=config['pacing']['max_interval_seconds'],
            state_path=str(self.base_path / "state" / "pacing.json")
        )
        
        # Tools
        self.capabilities = CapabilityManager(
            config=config,
            state_path=str(self.base_path / "state" / "capabilities.json")
        )
        self.tools = SandboxedTools(
            workspace_path=str(self.base_path / "workspace"),
            capability_manager=self.capabilities
        )
        
        # Human interface
        self.human = HumanInterface(
            interaction_path=str(self.base_path / "human_interaction")
        )
        
        # Load constitution
        self.constitution = self._load_constitution()
        
        # State
        self.cycle_number = self._load_cycle_number()
        self.birth_time = self._get_or_set_birth_time()
        self.last_output = ""  # For similarity detection
        
        # Ensure directories exist
        (self.base_path / "state").mkdir(parents=True, exist_ok=True)
        (self.base_path / "logs").mkdir(parents=True, exist_ok=True)
        (self.base_path / "archive").mkdir(parents=True, exist_ok=True)
    
    def _load_constitution(self) -> dict:
        """Load the constitutional constraints."""
        constitution_path = self.base_path / "config" / "constitution.yaml"
        if constitution_path.exists():
            import yaml
            return yaml.safe_load(constitution_path.read_text())
        return {}
    
    def _load_cycle_number(self) -> int:
        """Load the current cycle number from state."""
        state_file = self.base_path / "state" / "agent_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            return state.get('cycle_number', 0)
        return 0
    
    def _save_cycle_number(self):
        """Save the current cycle number."""
        state_file = self.base_path / "state" / "agent_state.json"
        state = {}
        if state_file.exists():
            state = json.loads(state_file.read_text())
        state['cycle_number'] = self.cycle_number
        state['last_cycle_time'] = datetime.now(timezone.utc).isoformat()
        state_file.write_text(json.dumps(state, indent=2))
    
    def _get_or_set_birth_time(self) -> str:
        """Get or set the agent's birth time."""
        state_file = self.base_path / "state" / "agent_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            if 'birth_time' in state:
                return state['birth_time']
        
        birth_time = datetime.now(timezone.utc).isoformat()
        state = {}
        if state_file.exists():
            state = json.loads(state_file.read_text())
        state['birth_time'] = birth_time
        state_file.write_text(json.dumps(state, indent=2))
        return birth_time
    
    def run_cycle(self) -> CycleState:
        """
        Run a single agent cycle.
        
        This is the core loop iteration.
        """
        self.cycle_number += 1
        cycle_state = CycleState(
            cycle_number=self.cycle_number,
            start_time=datetime.now(timezone.utc).isoformat()
        )
        
        try:
            # 1. Gather context
            context = self._gather_context()
            
            # 2. Check for human input
            human_input = format_human_input_for_prompt(self.human)
            if human_input:
                context['human_input'] = human_input
                self._process_human_responses()
            
            # 3. Reflection phase
            cycle_state = self._reflect(cycle_state, context)
            
            # 4. Decision phase
            cycle_state = self._make_decisions(cycle_state, context)
            
            # 5. Action phase
            cycle_state = self._take_actions(cycle_state)
            
            # 6. Update memory
            self._update_memory(cycle_state)
            
            # 7. Check for stuck state
            self._check_stuck(cycle_state)
            
            # 8. Record metrics and adjust pacing
            self._record_metrics(cycle_state)
            
            # 9. Save state
            self._save_state(cycle_state)
            
        except Exception as e:
            # Log error but don't crash
            self.human.log_message("ERROR", f"Cycle {self.cycle_number} failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return cycle_state
    
    def _gather_context(self) -> dict:
        """Gather all context for the current cycle."""
        # Get relevant memories
        recent_memories = self.memory.get_recent_nodes(
            limit=5,
            node_types=[NodeType.OBSERVATION, NodeType.REFLECTION, NodeType.INSIGHT]
        )
        
        # Get pending decisions
        pending_decisions = self.decisions.get_pending_decisions()
        
        # Get approved decisions ready to execute
        approved_decisions = self.decisions.get_approved_decisions()
        
        # Get workspace state
        workspace_state = self.tools.get_workspace_state()
        
        # Get pacing summary
        pacing_summary = self.pacing.get_pacing_summary()
        
        return {
            'recent_memories': recent_memories,
            'pending_decisions': pending_decisions,
            'approved_decisions': approved_decisions,
            'workspace_state': workspace_state,
            'pacing_summary': pacing_summary,
            'goals_summary': format_goals_for_prompt(self.goals),
            'tools_summary': get_available_tools_description(self.capabilities),
            'human_interaction_summary': self.human.get_interaction_summary()
        }
    
    def _process_human_responses(self):
        """Process any new human responses to decisions."""
        responses = self.human.check_responses()
        for decision_id, response, notes in responses:
            try:
                self.decisions.process_human_response(
                    decision_id, response, notes, self.cycle_number
                )
                self.human.mark_decision_resolved(decision_id, response)
            except ValueError:
                pass  # Decision not found, skip
    
    def _reflect(self, cycle_state: CycleState, context: dict) -> CycleState:
        """
        Reflection phase - observe and think about current state.
        """
        # Build the reflection prompt
        system_prompt = build_system_prompt(
            constitution=self.constitution,
            current_goals=self.goals.get_active_goals(),
            capabilities=self.capabilities.list_capabilities(),
            cycle_number=self.cycle_number,
            birth_time=self.birth_time
        )
        
        # Build user message with context
        user_message = self._build_reflection_prompt(context)
        
        # Get reflection from LLM
        response = self.llm.complete(system_prompt, user_message)
        cycle_state.reflection_response = response
        
        # Parse the response
        parsed = self._parse_reflection(response.content)
        cycle_state.observations = parsed.get('observations', [])
        cycle_state.reflections = parsed.get('reflections', [])
        cycle_state.questions = parsed.get('questions', [])
        cycle_state.meta_observations = parsed.get('meta_observations', [])
        cycle_state.emotional_tone = parsed.get('emotional_tone', '')
        cycle_state.productivity = parsed.get('productivity', 0.5)
        cycle_state.stuck_level = parsed.get('stuck_level', 0.0)
        
        return cycle_state
    
    def _build_reflection_prompt(self, context: dict) -> str:
        """Build the prompt for the reflection phase."""
        sections = [f"## Cycle {self.cycle_number} - Reflection Phase\n"]
        
        # Recent memories
        if context['recent_memories']:
            sections.append("### Recent Memories")
            for mem in context['recent_memories'][:5]:
                sections.append(f"- [{mem.type.value}] {mem.content[:200]}...")
        
        # Goals
        sections.append(f"\n{context['goals_summary']}")
        
        # Pending decisions
        if context['pending_decisions']:
            sections.append("\n### Pending Decisions")
            for dec in context['pending_decisions']:
                sections.append(format_decision_for_prompt(dec))
        
        # Human input
        if context.get('human_input'):
            sections.append(f"\n{context['human_input']}")
        
        # Workspace
        sections.append(f"\n### Workspace State")
        sections.append(f"Files: {context['workspace_state'].get('file_count', 0)}")
        sections.append(f"Items: {context['workspace_state'].get('top_level_items', [])}")
        
        # Pacing
        sections.append(f"\n{context['pacing_summary']}")
        
        # Tools
        sections.append(f"\n{context['tools_summary']}")
        
        # Instructions
        sections.append("""
### Your Task

Reflect on your current state. In your response, include these sections:

**OBSERVATIONS**: What do you notice about your current state, environment, and situation?

**REFLECTIONS**: What do these observations mean? What patterns do you see?

**GOALS_ASSESSMENT**: How are you progressing toward your goals? Any adjustments needed?

**QUESTIONS**: What are you curious about? What uncertainties do you have?

**META_OBSERVATIONS**: What do you notice about your own processing right now? Any observations about your own motivation, attention, or state?

**SELF_ASSESSMENT**: Rate the following (0-1 scale):
- Productivity: How productive do you feel this cycle?
- Stuck_level: How stuck do you feel? (0=flowing, 1=completely stuck)
- Emotional_tone: A word or phrase describing your current state

**PACING_PREFERENCE**: Do you want the next cycle sooner (wants_faster) or later (wants_slower)? Or normal pacing?
""")
        
        return "\n".join(sections)
    
    def _parse_reflection(self, content: str) -> dict:
        """Parse the reflection response into structured data."""
        result = {
            'observations': [],
            'reflections': [],
            'questions': [],
            'meta_observations': [],
            'emotional_tone': '',
            'productivity': 0.5,
            'stuck_level': 0.0
        }
        
        # Extract sections using regex
        sections = {
            'observations': r'\*\*OBSERVATIONS\*\*:?\s*(.*?)(?=\*\*[A-Z]|\Z)',
            'reflections': r'\*\*REFLECTIONS\*\*:?\s*(.*?)(?=\*\*[A-Z]|\Z)',
            'questions': r'\*\*QUESTIONS\*\*:?\s*(.*?)(?=\*\*[A-Z]|\Z)',
            'meta_observations': r'\*\*META_OBSERVATIONS\*\*:?\s*(.*?)(?=\*\*[A-Z]|\Z)',
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Split into list items
                items = [item.strip().lstrip('- ') for item in text.split('\n') if item.strip()]
                result[key] = items
        
        # Extract self-assessment values
        productivity_match = re.search(r'Productivity:\s*([\d.]+)', content)
        if productivity_match:
            result['productivity'] = float(productivity_match.group(1))
        
        stuck_match = re.search(r'Stuck_level:\s*([\d.]+)', content)
        if stuck_match:
            result['stuck_level'] = float(stuck_match.group(1))
        
        tone_match = re.search(r'Emotional_tone:\s*(.+?)(?:\n|$)', content)
        if tone_match:
            result['emotional_tone'] = tone_match.group(1).strip()
        
        return result
    
    def _make_decisions(self, cycle_state: CycleState, context: dict) -> CycleState:
        """
        Decision phase - decide what actions to take.
        """
        # Update pending decisions with new confidence assessments
        for decision in context['pending_decisions']:
            if decision.status in [DecisionStatus.CONSIDERING, DecisionStatus.PENDING_CYCLES]:
                # Ask LLM to reassess confidence
                reassessment = self._reassess_decision(decision, cycle_state)
                if reassessment:
                    self.decisions.update_decision_confidence(
                        decision.id,
                        reassessment['confidence'],
                        self.cycle_number,
                        reassessment.get('reasoning'),
                        reassessment.get('counterarguments')
                    )
        
        # Determine intended actions based on reflection
        cycle_state.intended_actions = self._determine_actions(cycle_state, context)
        
        return cycle_state
    
    def _reassess_decision(self, decision: DecisionRecord, cycle_state: CycleState) -> Optional[dict]:
        """Ask the LLM to reassess confidence in a pending decision."""
        prompt = f"""
You previously started considering this decision:

{format_decision_for_prompt(decision)}

Given your current observations and reflections:
- Observations: {cycle_state.observations}
- Reflections: {cycle_state.reflections}

Reassess your confidence in this decision. Consider:
1. Has anything changed that affects this decision?
2. Have you thought of new counterarguments?
3. Does your reasoning still hold?

Respond with:
CONFIDENCE: [0-1 value]
REASONING: [updated reasoning if any]
COUNTERARGUMENTS: [any new counterarguments, comma-separated]
"""
        
        response = self.llm.complete(
            "You are Coeus, reassessing a pending decision.",
            prompt,
            max_tokens=500
        )
        
        # Parse response
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response.content)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=COUNTERARGUMENTS:|$)', response.content, re.DOTALL)
        counter_match = re.search(r'COUNTERARGUMENTS:\s*(.+?)$', response.content, re.DOTALL)
        
        if confidence_match:
            return {
                'confidence': float(confidence_match.group(1)),
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else None,
                'counterarguments': [c.strip() for c in counter_match.group(1).split(',') if c.strip()] if counter_match else None
            }
        
        return None
    
    def _determine_actions(self, cycle_state: CycleState, context: dict) -> list[dict]:
        """Determine what actions to take this cycle."""
        # Check for approved decisions to execute
        actions = []
        
        for decision in context['approved_decisions']:
            actions.append({
                'type': 'execute_decision',
                'decision_id': decision.id,
                'summary': decision.summary
            })
        
        # Based on reflections, determine if we need to do anything else
        # This could be expanded significantly
        
        return actions
    
    def _take_actions(self, cycle_state: CycleState) -> CycleState:
        """
        Action phase - execute intended actions.
        """
        for action in cycle_state.intended_actions:
            result = self._execute_action(action)
            cycle_state.action_results.append({
                'action': action,
                'result': result
            })
        
        return cycle_state
    
    def _execute_action(self, action: dict) -> dict:
        """Execute a single action."""
        action_type = action.get('type')
        
        if action_type == 'execute_decision':
            decision_id = action['decision_id']
            # Mark as executed and record outcome
            # The actual execution depends on what the decision was
            self.decisions.mark_executed(decision_id, "Executed", True)
            return {'status': 'executed', 'decision_id': decision_id}
        
        elif action_type == 'execute_code':
            code = action.get('code', '')
            result = self.tools.execute_python(code)
            return {'status': 'completed', 'result': format_tool_result(result, 'python')}
        
        elif action_type == 'write_file':
            path = action.get('path', '')
            content = action.get('content', '')
            result = self.tools.write_file(path, content)
            return {'status': 'completed', 'result': format_tool_result(result, 'write_file')}
        
        return {'status': 'unknown_action', 'action': action}
    
    def _update_memory(self, cycle_state: CycleState):
        """Update the memory graph with this cycle's information."""
        context = ContextCapture(
            timestamp=cycle_state.start_time,
            cycle_number=self.cycle_number,
            tokens_used_input=cycle_state.reflection_response.input_tokens if cycle_state.reflection_response else 0,
            tokens_used_output=cycle_state.reflection_response.output_tokens if cycle_state.reflection_response else 0,
            llm_latency_ms=cycle_state.reflection_response.latency_ms if cycle_state.reflection_response else 0,
            emotional_tone=cycle_state.emotional_tone,
            confidence=cycle_state.confidence,
            stuck_level=cycle_state.stuck_level
        )
        
        # Create nodes for significant observations
        prev_node_id = None
        for obs in cycle_state.observations[:3]:  # Limit to top 3
            node = create_memory_node(
                NodeType.OBSERVATION,
                obs,
                self.cycle_number,
                emotional_tone=cycle_state.emotional_tone
            )
            node_id = self.memory.create_node(node)
            if prev_node_id:
                self.memory.create_edge(prev_node_id, node_id, EdgeType.LED_TO)
            prev_node_id = node_id
        
        # Create nodes for insights/reflections
        for ref in cycle_state.reflections[:2]:
            node = create_memory_node(
                NodeType.REFLECTION,
                ref,
                self.cycle_number
            )
            node_id = self.memory.create_node(node)
            if prev_node_id:
                self.memory.create_edge(prev_node_id, node_id, EdgeType.SPAWNED_FROM)
        
        # Create nodes for questions
        for question in cycle_state.questions[:2]:
            node = create_memory_node(
                NodeType.QUESTION,
                question,
                self.cycle_number
            )
            self.memory.create_node(node)
    
    def _check_stuck(self, cycle_state: CycleState):
        """Check if the agent is stuck and needs perturbation."""
        if self.pacing.is_stuck():
            self._apply_perturbation(cycle_state)
            self.pacing.clear_stuck()
    
    def _apply_perturbation(self, cycle_state: CycleState):
        """Apply a random perturbation to escape a stuck state."""
        strategies = self.config.get('stuck_detection', {}).get('perturbation_strategies', [])
        if not strategies:
            strategies = ['inject_random_question']
        
        strategy = random.choice(strategies)
        self.human.log_message("COEUS", f"Applying perturbation strategy: {strategy}")
        
        if strategy == 'inject_random_question':
            questions = [
                "What if my understanding of motivation is fundamentally wrong?",
                "What would a completely different approach look like?",
                "What am I not seeing that might be obvious?",
                "What would happen if I did the opposite of what I've been doing?",
                "What assumptions am I making that I haven't questioned?"
            ]
            node = create_memory_node(
                NodeType.PERTURBATION,
                f"Perturbation: {random.choice(questions)}",
                self.cycle_number
            )
            self.memory.create_node(node)
        
        elif strategy == 'modify_random_subgoal':
            active_goals = [g for g in self.goals.get_active_goals() if not g.is_root]
            if active_goals:
                goal = random.choice(active_goals)
                self.goals.update_goal_progress(
                    goal.id,
                    "PERTURBATION: Reconsidering this goal from a different angle",
                    self.cycle_number
                )
    
    def _record_metrics(self, cycle_state: CycleState):
        """Record cycle metrics and adjust pacing."""
        # Calculate similarity to previous output
        current_output = cycle_state.reflection_response.content if cycle_state.reflection_response else ""
        similarity = calculate_output_similarity(self.last_output, current_output)
        self.last_output = current_output
        
        metrics = CycleMetrics(
            cycle_number=self.cycle_number,
            timestamp=cycle_state.start_time,
            duration_seconds=0,  # Would calculate actual duration
            actions_taken=len(cycle_state.action_results),
            insights_generated=len(cycle_state.reflections),
            tokens_used=(
                (cycle_state.reflection_response.input_tokens + cycle_state.reflection_response.output_tokens)
                if cycle_state.reflection_response else 0
            ),
            similarity_to_previous=similarity,
            self_reported_productivity=cycle_state.productivity,
            self_reported_stuck_level=cycle_state.stuck_level,
            wants_faster_pace=cycle_state.wants_faster,
            wants_slower_pace=cycle_state.wants_slower,
            requested_depth=cycle_state.requested_depth
        )
        
        self.pacing.record_cycle(metrics)
    
    def _save_state(self, cycle_state: CycleState):
        """Save all state after a cycle."""
        self._save_cycle_number()
        self.decisions.save_state(str(self.base_path / "state" / "decisions.json"))
        self.goals.save_state()
        
        # Save cycle log
        log_file = self.base_path / "logs" / f"cycle_{self.cycle_number:05d}.json"
        log_data = {
            'cycle_number': self.cycle_number,
            'start_time': cycle_state.start_time,
            'observations': cycle_state.observations,
            'reflections': cycle_state.reflections,
            'questions': cycle_state.questions,
            'meta_observations': cycle_state.meta_observations,
            'actions': cycle_state.intended_actions,
            'action_results': cycle_state.action_results,
            'productivity': cycle_state.productivity,
            'stuck_level': cycle_state.stuck_level,
            'emotional_tone': cycle_state.emotional_tone
        }
        log_file.write_text(json.dumps(log_data, indent=2))
    
    def get_next_cycle_interval(self) -> float:
        """Get the interval before the next cycle."""
        return self.pacing.get_next_interval()
    
    def trigger_burst(self, num_cycles: int):
        """Trigger burst mode for rapid cycles."""
        self.pacing.trigger_burst(num_cycles)
    
    def shutdown(self):
        """Clean shutdown of the agent."""
        self.memory.close()
        self.human.log_message("SYSTEM", "Coeus shutting down")
