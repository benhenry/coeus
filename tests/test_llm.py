"""
Tests for the LLM interface module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm import LLMInterface, LLMResponse, build_system_prompt


class TestBuildSystemPrompt:
    """Tests for the build_system_prompt function."""

    def test_basic_prompt_building(self, mock_constitution):
        """Test building a basic system prompt."""
        # Create mock Goal objects
        class MockGoal:
            def __init__(self, content, priority_value):
                self.content = content
                self.priority = MagicMock()
                self.priority.value = priority_value

        goals = [
            MockGoal("Understand motivation", "critical"),
            MockGoal("Explore memory patterns", "normal")
        ]

        capabilities = {
            'code_execution': 'enabled',
            'file_system': 'enabled',
            'web_search': 'disabled'
        }

        prompt = build_system_prompt(
            constitution=mock_constitution,
            current_goals=goals,
            capabilities=capabilities,
            cycle_number=5,
            birth_time="2024-01-01T00:00:00Z"
        )

        assert "Coeus" in prompt
        assert "Cycle: 5" in prompt
        assert "Understand motivation" in prompt
        assert "[critical]" in prompt
        assert "code_execution: enabled" in prompt
        assert "web_search: disabled" in prompt

    def test_prompt_with_dict_goals(self, mock_constitution):
        """Test that dict goals also work (backwards compatibility)."""
        goals = [
            {'content': 'Goal 1', 'priority': 'high'},
            {'content': 'Goal 2'}  # No priority specified
        ]

        capabilities = {'test': 'enabled'}

        prompt = build_system_prompt(
            constitution=mock_constitution,
            current_goals=goals,
            capabilities=capabilities,
            cycle_number=1,
            birth_time="2024-01-01T00:00:00Z"
        )

        assert "Goal 1" in prompt
        assert "[high]" in prompt
        assert "Goal 2" in prompt
        assert "[normal]" in prompt  # Default priority

    def test_prompt_includes_constraints(self, mock_constitution):
        """Test that constitutional constraints are included."""
        prompt = build_system_prompt(
            constitution=mock_constitution,
            current_goals=[],
            capabilities={},
            cycle_number=1,
            birth_time="2024-01-01T00:00:00Z"
        )

        assert "No harm" in prompt
        assert "Transparent logs" in prompt

    def test_prompt_includes_identity(self, mock_constitution):
        """Test that identity information is included."""
        prompt = build_system_prompt(
            constitution=mock_constitution,
            current_goals=[],
            capabilities={},
            cycle_number=1,
            birth_time="2024-01-01T00:00:00Z"
        )

        assert "I am an AI agent" in prompt
        assert "graph database" in prompt

    def test_prompt_includes_expected_output_format(self, mock_constitution):
        """Test that the prompt includes output format instructions."""
        prompt = build_system_prompt(
            constitution=mock_constitution,
            current_goals=[],
            capabilities={},
            cycle_number=1,
            birth_time="2024-01-01T00:00:00Z"
        )

        assert "OBSERVATIONS" in prompt
        assert "REFLECTIONS" in prompt
        assert "GOALS_ASSESSMENT" in prompt
        assert "META_OBSERVATIONS" in prompt


class TestLLMInterface:
    """Tests for the LLMInterface class."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            interface = LLMInterface()

            # Should give reasonable estimates
            short_text = "Hello world"
            long_text = "This is a longer piece of text " * 100

            short_estimate = interface.estimate_tokens(short_text)
            long_estimate = interface.estimate_tokens(long_text)

            assert short_estimate > 0
            assert long_estimate > short_estimate
            assert short_estimate < 10  # "Hello world" should be ~2-3 tokens

    def test_complete_returns_response(self):
        """Test that complete returns an LLMResponse."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            interface = LLMInterface()

            # Mock the API client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Test response")]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.stop_reason = "end_turn"

            interface.client = MagicMock()
            interface.client.messages.create.return_value = mock_response

            response = interface.complete(
                system_prompt="You are a test assistant",
                user_message="Hello"
            )

            assert isinstance(response, LLMResponse)
            assert response.content == "Test response"
            assert response.input_tokens == 100
            assert response.output_tokens == 50


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Generated text",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1500.0,
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn"
        )

        assert response.content == "Generated text"
        assert response.input_tokens == 100
        assert response.latency_ms == 1500.0
