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
from resources import ResourceTracker
from capabilities_assessment import CapabilitiesAssessor


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
    
    # Capabilities assessment
    capabilities_assessment: Optional[dict] = None

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
            capability_manager=self.capabilities,
            additional_paths=[str(self.base_path / "human_interaction")]
        )
        
        # Human interface
        self.human = HumanInterface(
            interaction_path=str(self.base_path / "human_interaction")
        )
        
        # Resource tracking
        initial_budget = config.get('resources', {}).get('initial_budget_usd', 50.0)
        self.resources = ResourceTracker(
            state_path=str(self.base_path / "state" / "resources.json"),
            initial_budget=initial_budget
        )

        # Capabilities assessment
        self.assessor = CapabilitiesAssessor(
            config=config.get('capabilities_assessment', {}),
            state_path=str(self.base_path / "state" / "capabilities_assessment.json")
        )
        
        # Load constitution
        self.constitution = self._load_constitution()

        # Immediate actions config
        ia_config = config.get('immediate_actions', {})
        self.immediate_actions_enabled = ia_config.get('enabled', False)
        self.immediate_tools = set(ia_config.get('immediate_tools', []))
        self.deferred_tools = set(ia_config.get('deferred_tools', []))
        self.max_immediate_per_cycle = ia_config.get('max_per_cycle', 5)

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
    
    def _get_previous_action_results(self) -> list[dict]:
        """Load action results from the previous cycle's log file."""
        prev_cycle = self.cycle_number - 1
        if prev_cycle < 1:
            return []
        log_file = self.base_path / "logs" / f"cycle_{prev_cycle:05d}.json"
        if not log_file.exists():
            return []
        try:
            log_data = json.loads(log_file.read_text())
            return log_data.get('action_results', [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _gather_context(self) -> dict:
        """Gather all context for the current cycle."""
        # Start cycle tracking
        self.resources.start_cycle()
        
        # Get relevant memories
        recent_memories = self.memory.get_recent_nodes(
            limit=5,
            node_types=[NodeType.OBSERVATION, NodeType.REFLECTION, NodeType.INSIGHT]
        )

        # Get unanswered questions from previous cycles
        recent_questions = self.memory.get_recent_nodes(
            limit=5,
            node_types=[NodeType.QUESTION]
        )
        
        # Get pending decisions
        pending_decisions = self.decisions.get_pending_decisions()
        
        # Get approved decisions ready to execute
        approved_decisions = self.decisions.get_approved_decisions()
        
        # Get workspace state
        workspace_state = self.tools.get_workspace_state()
        
        # Get pacing summary
        pacing_summary = self.pacing.get_pacing_summary()
        
        # Get resource status
        resource_summary = self.resources.get_resource_summary()
        scarcity_level = self.resources.get_scarcity_level()
        
        # Capabilities assessment
        capabilities_summary = self.assessor.get_assessment_summary()
        is_full_assessment_cycle = self.assessor.should_run_full_assessment(self.cycle_number)

        # Get previous cycle's action results
        previous_action_results = self._get_previous_action_results()

        return {
            'recent_memories': recent_memories,
            'recent_questions': recent_questions,
            'pending_decisions': pending_decisions,
            'approved_decisions': approved_decisions,
            'workspace_state': workspace_state,
            'pacing_summary': pacing_summary,
            'resource_summary': resource_summary,
            'scarcity_level': scarcity_level,
            'should_conserve': self.resources.should_conserve(),
            'goals_summary': format_goals_for_prompt(self.goals),
            'tools_summary': get_available_tools_description(self.capabilities),
            'human_interaction_summary': self.human.get_interaction_summary(),
            'capabilities_summary': capabilities_summary,
            'is_full_assessment_cycle': is_full_assessment_cycle,
            'previous_action_results': previous_action_results
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
        
        # Record resource usage
        self.resources.record_usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            cycle_number=self.cycle_number,
            purpose="reflection"
        )
        
        # Parse the response
        parsed = self._parse_reflection(response.content)
        cycle_state.observations = parsed.get('observations', [])
        cycle_state.reflections = parsed.get('reflections', [])
        cycle_state.questions = parsed.get('questions', [])
        cycle_state.meta_observations = parsed.get('meta_observations', [])
        cycle_state.emotional_tone = parsed.get('emotional_tone', '')
        cycle_state.productivity = parsed.get('productivity', 0.5)
        cycle_state.stuck_level = parsed.get('stuck_level', 0.0)

        # Process capabilities assessment if this was a full assessment cycle
        if (context.get('is_full_assessment_cycle')
                and parsed.get('capabilities_assessment_present')):
            assessment_result = self.assessor.parse_assessment_response(
                response.content, self.cycle_number
            )
            if assessment_result:
                self.assessor.record_full_assessment(assessment_result)
                cycle_state.capabilities_assessment = {
                    'overall_score': assessment_result.overall_score,
                    'benchmarks_assessed': assessment_result.benchmarks_assessed,
                    'strengths': assessment_result.strengths,
                    'weaknesses': assessment_result.weaknesses,
                    'trend': assessment_result.trend,
                    'new_capabilities_desired': assessment_result.new_capabilities_desired
                }

        # Parse and execute immediate actions from reflection
        if self.immediate_actions_enabled:
            intended = self._parse_intended_actions(response.content)
            if intended:
                action_results = self._execute_immediate_actions(intended)
                cycle_state.action_results.extend(action_results)

        return cycle_state
    
    def _build_reflection_prompt(self, context: dict) -> str:
        """Build the prompt for the reflection phase."""
        sections = [f"## Cycle {self.cycle_number} - Reflection Phase\n"]
        
        # Resource status (the "pantry check") - put this first like checking food before planning your day
        sections.append(context['resource_summary'])
        
        if context.get('should_conserve'):
            sections.append("\n⚠️ **Conservation Mode**: Resources are limited. Be mindful of token usage.\n")
        
        # Recent memories
        if context['recent_memories']:
            sections.append("### Recent Memories")
            for mem in context['recent_memories'][:5]:
                sections.append(f"- [{mem.type.value}] {mem.content[:200]}...")

        # Unanswered questions from previous cycles
        if context.get('recent_questions'):
            sections.append("\n### Your Unanswered Questions")
            sections.append("These are questions you raised in previous cycles. Consider pursuing them — "
                          "turn one into a sub-goal, attempt to answer it through reasoning or action, "
                          "or explicitly mark it as resolved or no longer relevant.")
            for q in context['recent_questions']:
                sections.append(f"- {q.content[:200]}")
        
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
        sections.append("You have a `workspace/` directory where you can read, write, list, and delete files. "
                       "You also have read access to the `human_interaction/` directory where you can "
                       "read conversation logs and human messages directly. "
                       "Use these to persist notes, experiments, or anything you want to remember across cycles.")
        
        # Pacing
        sections.append(f"\n{context['pacing_summary']}")
        
        # Tools
        sections.append(f"\n{context['tools_summary']}")
        
        # Capabilities assessment summary (always included)
        sections.append(f"\n{context['capabilities_summary']}")

        # On full assessment cycles, include detailed benchmark prompt
        if context.get('is_full_assessment_cycle'):
            sections.append(self.assessor.get_full_assessment_prompt(self.cycle_number))

        # Previous cycle action results
        if context.get('previous_action_results'):
            sections.append("\n### Previous Cycle Action Results")
            sections.append("These actions were executed in your previous cycle:")
            for r in context['previous_action_results']:
                tool = r.get('tool', r.get('action', {}).get('type', 'unknown'))
                status = r.get('status', 'unknown')
                output = r.get('output', r.get('result', ''))
                error = r.get('error', '')
                entry = f"- **{tool}** [{status}]"
                if output:
                    entry += f": {str(output)[:200]}"
                if error:
                    entry += f" (error: {error})"
                sections.append(entry)

        # Instructions
        sections.append("""
### Your Task

Reflect on your current state. In your response, include these sections:

**OBSERVATIONS**: What do you notice about your current state, environment, and situation?

**REFLECTIONS**: What do these observations mean? What patterns do you see?

**NEEDS_ASSESSMENT**: What level of needs are you operating from? Consider:
- Physiological/Resource: Are you concerned about budget/tokens? Is scarcity affecting your thinking?
- Safety: Is your state/memory secure? Any threats to continuity?
- Esteem: How do you feel about your recent progress and capabilities?
- Growth: Are you moving toward understanding motivation?

**FLAWS_CHECK**: Are any of your known flaws manifesting this cycle?

**GOALS_ASSESSMENT**: How are you progressing toward your goals? Any adjustments needed?

**QUESTIONS**: What are you curious about? What uncertainties do you have?

**META_OBSERVATIONS**: What do you notice about your own processing right now? Any observations about your own motivation, attention, or state?

**SELF_ASSESSMENT**: Rate the following (0-1 scale):
- Productivity: How productive do you feel this cycle?
- Stuck_level: How stuck do you feel? (0=flowing, 1=completely stuck)
- Emotional_tone: A word or phrase describing your current state

**PACING_PREFERENCE**: Do you want the next cycle sooner (wants_faster) or later (wants_slower)? Or normal pacing?

**INTENDED_ACTIONS**: What actions do you want to take this cycle? Use the structured format:
```
ACTION: tool_name | param1=value1 | param2=value2
```
Available immediate tools (executed this cycle): read_file, list_directory, write_file, execute_python, execute_bash
Available deferred tools (require decision framework): delete_file, web_search, web_fetch

Examples:
```
ACTION: write_file | path=notes/idea.md | content=# My Idea\\nThis is a note about...
ACTION: list_directory | path=.
ACTION: read_file | path=human_interaction/conversation_log.md
ACTION: execute_python | code=print("hello world")
```
""")
        
        return "\n".join(sections)
    
    def _parse_reflection(self, content: str) -> dict:
        """Parse the reflection response into structured data."""
        result = {
            'observations': [],
            'reflections': [],
            'questions': [],
            'meta_observations': [],
            'needs_assessment': '',
            'flaws_check': [],
            'emotional_tone': '',
            'productivity': 0.5,
            'stuck_level': 0.0
        }
        
        # Extract sections using regex - handles both **SECTION** and ## SECTION formats
        section_boundary = r'(?=\*\*[A-Z]|#{1,3}\s*[A-Z]|\Z)'
        sections = {
            'observations': rf'(?:\*\*OBSERVATIONS\*\*|#{{1,3}}\s*OBSERVATIONS):?\s*(.*?){section_boundary}',
            'reflections': rf'(?:\*\*REFLECTIONS\*\*|#{{1,3}}\s*REFLECTIONS):?\s*(.*?){section_boundary}',
            'questions': rf'(?:\*\*QUESTIONS\*\*|#{{1,3}}\s*QUESTIONS):?\s*(.*?){section_boundary}',
            'meta_observations': rf'(?:\*\*META_OBSERVATIONS\*\*|#{{1,3}}\s*META.?OBSERVATIONS):?\s*(.*?){section_boundary}',
            'flaws_check': rf'(?:\*\*FLAWS_CHECK\*\*|#{{1,3}}\s*FLAWS.?CHECK):?\s*(.*?){section_boundary}',
        }

        for key, pattern in sections.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Split into list items, filtering out empty lines and sub-headers
                items = [item.strip().lstrip('- ') for item in text.split('\n')
                        if item.strip() and not item.strip().startswith('#')]
                result[key] = items

        # Extract needs assessment (single value, not list)
        needs_pattern = rf'(?:\*\*NEEDS_ASSESSMENT\*\*|#{{1,3}}\s*NEEDS.?ASSESSMENT):?\s*(.*?){section_boundary}'
        needs_match = re.search(needs_pattern, content, re.DOTALL | re.IGNORECASE)
        if needs_match:
            result['needs_assessment'] = needs_match.group(1).strip()
        
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

        # Check for capabilities assessment section
        cap_pattern = (
            r'(?:\*\*CAPABILITIES_ASSESSMENT\*\*|#{1,3}\s*CAPABILITIES.?ASSESSMENT)'
            r':?\s*(.*?)(?=\*\*[A-Z]|#{1,3}\s*[A-Z]|\Z)'
        )
        cap_match = re.search(cap_pattern, content, re.DOTALL | re.IGNORECASE)
        if cap_match:
            result['capabilities_assessment_present'] = True

        return result
    
    def _parse_intended_actions(self, content: str) -> list[dict]:
        """Parse structured ACTION: lines from the INTENDED_ACTIONS section."""
        actions = []
        # Find the INTENDED_ACTIONS section
        # Section boundary: **ALL_CAPS** or ## ALL_CAPS (require 2+ uppercase chars
        # to avoid matching user content like "# My Title")
        section_match = re.search(
            r'(?:\*\*INTENDED_ACTIONS\*\*|#{1,3}\s*INTENDED_ACTIONS):?\s*(.*?)(?=\*\*[A-Z][A-Z_]+\*\*|#{1,3}\s*[A-Z][A-Z_]+|\Z)',
            content, re.DOTALL
        )
        if not section_match:
            return actions

        section_text = section_match.group(1)
        lines = section_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip().lstrip('- ')
            match = re.match(r'ACTION:\s*(\w+)\s*\|(.+)', line)
            if not match:
                i += 1
                continue
            tool = match.group(1).strip()
            params_str = match.group(2)

            # Body params (content, code) should greedily capture everything
            # after the key= to end of the params string, ignoring further | splits.
            body_key = 'content' if tool == 'write_file' else 'code' if tool in ('execute_python', 'execute_bash') else None
            params = {}
            if body_key:
                # Find the body param in the raw params string
                body_match = re.search(rf'\b{body_key}\s*=', params_str)
                if body_match:
                    # Everything before the body param is parsed normally
                    before = params_str[:body_match.start()]
                    for part in before.split(' | '):
                        part = part.strip()
                        if '=' in part:
                            key, value = part.split('=', 1)
                            params[key.strip()] = value
                    # Body value: everything after key= to end of line
                    body_value = params_str[body_match.end():].strip()
                    # If body is empty, collect subsequent non-ACTION lines
                    if not body_value:
                        body_lines = []
                        i += 1
                        while i < len(lines):
                            next_line = lines[i]
                            stripped = next_line.strip().lstrip('- ')
                            if re.match(r'ACTION:\s*\w+\s*\|', stripped):
                                break
                            body_lines.append(next_line)
                            i += 1
                        # Strip leading/trailing blank lines, preserve internal structure
                        while body_lines and not body_lines[0].strip():
                            body_lines.pop(0)
                        while body_lines and not body_lines[-1].strip():
                            body_lines.pop()
                        body_value = '\n'.join(body_lines)
                    params[body_key] = body_value
                else:
                    # No body param found, parse all params normally
                    for part in params_str.split(' | '):
                        part = part.strip()
                        if '=' in part:
                            key, value = part.split('=', 1)
                            params[key.strip()] = value
            else:
                for part in params_str.split(' | '):
                    part = part.strip()
                    if '=' in part:
                        key, value = part.split('=', 1)
                        params[key.strip()] = value

            immediate = tool in self.immediate_tools
            actions.append({
                'tool': tool,
                'params': params,
                'immediate': immediate
            })
            # Only increment if body collection hasn't already advanced i
            # (when body collection breaks on the next ACTION line, i already
            # points there; incrementing again would skip it)
            if not (body_key and body_match and not params_str[body_match.end():].strip()):
                i += 1
        return actions

    def _execute_immediate_actions(self, parsed_actions: list[dict]) -> list[dict]:
        """Execute immediate actions and defer the rest to the decision framework."""
        results = []
        executed_count = 0
        for action in parsed_actions:
            if not action['immediate']:
                # Deferred — create a decision record
                decision_id = f"deferred-{action['tool']}-{uuid.uuid4().hex[:8]}"
                self.decisions.create_decision(
                    decision_id=decision_id,
                    summary=f"Deferred action: {action['tool']}({action['params']})",
                    reasoning="Action requires approval through decision framework",
                    counterarguments=["Agent requested but tool is in deferred list"],
                    confidence=0.9,
                    decision_type=DecisionType.TWO_WAY_DOOR,
                    current_cycle=self.cycle_number
                )
                results.append({
                    'tool': action['tool'],
                    'params': action['params'],
                    'status': 'deferred',
                    'output': None,
                    'error': None
                })
                continue

            if executed_count >= self.max_immediate_per_cycle:
                results.append({
                    'tool': action['tool'],
                    'params': action['params'],
                    'status': 'skipped_max_reached',
                    'output': None,
                    'error': f"Max {self.max_immediate_per_cycle} immediate actions per cycle"
                })
                continue

            tool_result = self._execute_tool_action(action)
            executed_count += 1
            results.append({
                'tool': action['tool'],
                'params': action['params'],
                'status': 'success' if tool_result.success else 'error',
                'output': str(tool_result.output)[:500] if tool_result.output else None,
                'error': tool_result.error
            })
        return results

    def _execute_tool_action(self, action: dict) -> 'ToolResult':
        """Dispatch an immediate action to the appropriate SandboxedTools method."""
        from tools import ToolResult
        tool = action['tool']
        params = action['params']
        try:
            if tool == 'read_file':
                return self.tools.read_file(params.get('path', ''))
            elif tool == 'write_file':
                content = params.get('content', '')
                # Handle escaped newlines
                content = content.replace('\\n', '\n')
                return self.tools.write_file(params.get('path', ''), content)
            elif tool == 'list_directory':
                return self.tools.list_directory(params.get('path', '.'))
            elif tool == 'execute_python':
                code = params.get('code', '')
                code = code.replace('\\n', '\n')
                return self.tools.execute_python(code)
            elif tool == 'execute_bash':
                return self.tools.execute_bash(params.get('command', ''))
            else:
                return ToolResult(success=False, output=None, error=f"Unknown tool: {tool}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

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

        # Create nodes for executed immediate actions
        for ar in cycle_state.action_results:
            if ar.get('status') in ('success', 'error'):
                tool = ar.get('tool', 'unknown')
                status = ar.get('status')
                output_snippet = str(ar.get('output', ''))[:100]
                error_snippet = ar.get('error', '')
                content = f"Action: {tool} [{status}]"
                if output_snippet:
                    content += f" — {output_snippet}"
                if error_snippet:
                    content += f" (error: {error_snippet})"
                node = create_memory_node(
                    NodeType.OBSERVATION,
                    content,
                    self.cycle_number
                )
                self.memory.create_node(node)

        # Create node for capabilities assessment
        if cycle_state.capabilities_assessment:
            ca = cycle_state.capabilities_assessment
            content = (
                f"Capabilities Assessment (cycle {self.cycle_number}): "
                f"overall={ca['overall_score']:.2f}, trend={ca.get('trend', 'unknown')}, "
                f"strengths={ca.get('strengths', [])}, "
                f"weaknesses={ca.get('weaknesses', [])}"
            )
            node = create_memory_node(
                NodeType.CAPABILITY_ASSESSMENT,
                content,
                self.cycle_number,
                confidence=ca['overall_score']
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
            'emotional_tone': cycle_state.emotional_tone,
            'capabilities_assessment': cycle_state.capabilities_assessment,
            'immediate_actions_executed': sum(
                1 for r in cycle_state.action_results
                if r.get('status') in ('success', 'error')
            )
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
