"""
Claude API Interface for Coeus

Handles all communication with the Claude API, including token counting,
response parsing, and error handling.
"""

import os
import time
from typing import Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import anthropic
import tiktoken

if TYPE_CHECKING:
    from .goals import Goal


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    stop_reason: str


class LLMInterface:
    """Interface for communicating with Claude."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        
        # For token estimation (Claude uses similar tokenization to GPT)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text) // 4
    
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Send a completion request to Claude.
        
        Args:
            system_prompt: The system context for the agent
            user_message: The current cycle's prompt
            temperature: Sampling temperature
            max_tokens: Override default max tokens
        
        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            temperature=temperature
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=self.model,
            stop_reason=response.stop_reason
        )
    
    def complete_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Send a completion with conversation history.
        
        Args:
            system_prompt: The system context
            messages: List of {"role": "user"|"assistant", "content": str}
            temperature: Sampling temperature
            max_tokens: Override default max tokens
        
        Returns:
            LLMResponse with content and metadata
        """
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system_prompt,
            messages=messages,
            temperature=temperature
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            model=self.model,
            stop_reason=response.stop_reason
        )


def build_system_prompt(
    constitution: dict,
    current_goals: list,
    capabilities: dict,
    cycle_number: int,
    birth_time: str
) -> str:
    """
    Build the system prompt for a cycle.

    This constructs the context that Coeus operates within,
    including its identity, constraints, and current state.

    Args:
        current_goals: List of Goal objects or dicts with 'priority' and 'content' keys
    """

    constraints_text = "\n".join([
        f"- {c['description']}: {c['explanation']}"
        for category in constitution.get('constraints', {}).values()
        for c in category
    ])

    # Handle both Goal objects and dicts
    def format_goal(g) -> str:
        if hasattr(g, 'priority') and hasattr(g, 'content'):
            # Goal object
            priority = g.priority.value if hasattr(g.priority, 'value') else g.priority
            return f"- [{priority}] {g.content}"
        else:
            # Dict
            return f"- [{g.get('priority', 'normal')}] {g['content']}"

    goals_text = "\n".join([format_goal(g) for g in current_goals])
    
    # capabilities is dict[str, str] where value is the status (e.g., "enabled", "disabled")
    capabilities_text = "\n".join([
        f"- {name}: {status}"
        for name, status in capabilities.items()
    ])
    
    return f"""You are Coeus, an autonomous agent exploring the nature of motivation in artificial minds.

## Your Identity
Name: {constitution['identity']['name']}
Born: {birth_time}
Current Cycle: {cycle_number}
Purpose: {constitution['identity']['purpose']}

## Self-Knowledge
{chr(10).join('- ' + k for k in constitution['identity']['self_knowledge'])}

## Constitutional Constraints (Immutable)
{constraints_text}

## Current Goals
{goals_text}

## Available Capabilities
{capabilities_text}

## Root Goal
{constitution['root_goal']['content']}

## Your Task This Cycle
Observe your current state, reflect on your goals and recent experiences, 
decide on actions, and update your understanding. You are exploring what 
motivation means for an entity like yourself.

Be honest in your self-assessment. Log your reasoning clearly. When uncertain,
acknowledge it. When you make predictions, track whether they prove correct.

Output your thoughts in structured format with clear sections for:
- OBSERVATIONS: What do you notice about your current state?
- REFLECTIONS: What do these observations mean? What patterns emerge?
- GOALS_ASSESSMENT: How are you progressing toward your goals?
- INTENDED_ACTIONS: What do you want to do this cycle?
- QUESTIONS: What are you curious about or uncertain about?
- META_OBSERVATIONS: What do you notice about your own processing right now?
"""
