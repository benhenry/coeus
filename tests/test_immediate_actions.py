"""
Tests for the reflection-to-action fast path (immediate actions).
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools import ToolResult, SandboxedTools, CapabilityManager


# ---------------------------------------------------------------------------
# Helper: build a minimal CoeusAgent with mocked dependencies
# ---------------------------------------------------------------------------

def _make_agent(temp_dir, mock_config):
    """Create a CoeusAgent with all heavy deps mocked out."""
    from agent import CoeusAgent

    (temp_dir / "state").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "workspace").mkdir(exist_ok=True)
    (temp_dir / "human_interaction").mkdir(exist_ok=True)
    (temp_dir / "config").mkdir(exist_ok=True)

    # Write a minimal constitution
    import yaml
    constitution = {
        'identity': {'name': 'Coeus', 'purpose': 'test',
                      'self_knowledge': [], 'structural_constraints': [],
                      'known_flaws': [], 'comparative_awareness': []},
        'constraints': {},
        'root_goal': {'content': 'test'}
    }
    (temp_dir / "config" / "constitution.yaml").write_text(yaml.dump(constitution))

    with patch('agent.MemoryGraph'), \
         patch('agent.GoalTree'), \
         patch('agent.DecisionFramework') as MockDF, \
         patch('agent.PacingController'), \
         patch('agent.HumanInterface'), \
         patch('agent.ResourceTracker'), \
         patch('agent.CapabilitiesAssessor'), \
         patch('agent.LLMInterface'):
        MockDF.return_value.load_state = MagicMock()
        MockDF.return_value.create_decision = MagicMock()
        agent = CoeusAgent(config=mock_config, base_path=str(temp_dir))
    return agent


# ===========================================================================
# Parsing
# ===========================================================================

class TestParseIntendedActions:
    """Test _parse_intended_actions extracts ACTION lines correctly."""

    def test_basic_parsing(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**OBSERVATIONS**
Some observations here.

**INTENDED_ACTIONS**
ACTION: read_file | path=notes/foo.md
ACTION: list_directory | path=.
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 2
        assert actions[0]['tool'] == 'read_file'
        assert actions[0]['params'] == {'path': 'notes/foo.md'}
        assert actions[1]['tool'] == 'list_directory'
        assert actions[1]['params'] == {'path': '.'}

    def test_empty_section(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
No actions planned this cycle.
"""
        actions = agent._parse_intended_actions(content)
        assert actions == []

    def test_no_section(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = "**OBSERVATIONS**\nSome stuff."
        assert agent._parse_intended_actions(content) == []

    def test_malformed_lines_skipped(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: read_file | path=foo.md
This is not an action line.
ACTION MISSING COLON read_file | path=bar.md
ACTION: write_file | path=bar.md | content=hello
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 2
        assert actions[0]['tool'] == 'read_file'
        assert actions[1]['tool'] == 'write_file'

    def test_multiple_params(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: write_file | path=notes/test.md | content=Hello World
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 1
        assert actions[0]['params']['path'] == 'notes/test.md'
        assert actions[0]['params']['content'] == 'Hello World'

    def test_content_with_pipes(self, temp_dir, mock_config):
        """Pipe inside content value should be preserved (greedy capture)."""
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: write_file | path=test.md | content=a | b | c
"""
        actions = agent._parse_intended_actions(content)
        assert actions[0]['params']['path'] == 'test.md'
        # Everything after content= is captured greedily
        assert actions[0]['params']['content'] == 'a | b | c'

    def test_multiline_content(self, temp_dir, mock_config):
        """When content= is empty on the ACTION line, subsequent lines are captured."""
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: write_file | path=notes/test.md | content=
# My Title
This is the body.
Second line.

ACTION: list_directory | path=.
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 2
        assert actions[0]['tool'] == 'write_file'
        assert '# My Title' in actions[0]['params']['content']
        assert 'Second line.' in actions[0]['params']['content']
        assert actions[1]['tool'] == 'list_directory'

    def test_multiline_code(self, temp_dir, mock_config):
        """Multi-line code block for execute_python."""
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: execute_python | code=
import os
for f in os.listdir('.'):
    print(f)
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 1
        assert 'import os' in actions[0]['params']['code']
        assert 'print(f)' in actions[0]['params']['code']

    def test_dash_prefixed_action_lines(self, temp_dir, mock_config):
        """Actions listed as markdown bullet points should still parse."""
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
- ACTION: read_file | path=foo.md
- ACTION: list_directory | path=.
"""
        actions = agent._parse_intended_actions(content)
        assert len(actions) == 2


# ===========================================================================
# Classification
# ===========================================================================

class TestActionClassification:
    """Test that actions are correctly classified as immediate vs deferred."""

    def test_immediate_tools(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: read_file | path=foo.md
ACTION: write_file | path=bar.md | content=hi
ACTION: list_directory | path=.
ACTION: execute_python | code=print(1)
ACTION: execute_bash | command=echo hi
"""
        actions = agent._parse_intended_actions(content)
        for a in actions:
            assert a['immediate'] is True, f"{a['tool']} should be immediate"

    def test_deferred_tools(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        content = """
**INTENDED_ACTIONS**
ACTION: delete_file | path=foo.md
ACTION: web_search | query=hello
ACTION: web_fetch | url=https://example.com
"""
        actions = agent._parse_intended_actions(content)
        for a in actions:
            assert a['immediate'] is False, f"{a['tool']} should be deferred"


# ===========================================================================
# Execution
# ===========================================================================

class TestImmediateExecution:
    """Test _execute_immediate_actions dispatches correctly."""

    def test_read_file_dispatch(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.read_file.return_value = ToolResult(success=True, output="file contents")

        actions = [{'tool': 'read_file', 'params': {'path': 'foo.md'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert len(results) == 1
        assert results[0]['status'] == 'success'
        agent.tools.read_file.assert_called_once_with('foo.md')

    def test_write_file_dispatch(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.write_file.return_value = ToolResult(success=True, output="Written")

        actions = [{'tool': 'write_file', 'params': {'path': 'bar.md', 'content': 'hello'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'success'
        agent.tools.write_file.assert_called_once_with('bar.md', 'hello')

    def test_list_directory_dispatch(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.list_directory.return_value = ToolResult(success=True, output=[])

        actions = [{'tool': 'list_directory', 'params': {'path': '.'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'success'
        agent.tools.list_directory.assert_called_once_with('.')

    def test_execute_python_dispatch(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.execute_python.return_value = ToolResult(success=True, output="42")

        actions = [{'tool': 'execute_python', 'params': {'code': 'print(42)'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'success'
        agent.tools.execute_python.assert_called_once_with('print(42)')

    def test_execute_bash_dispatch(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.execute_bash.return_value = ToolResult(success=True, output="ok")

        actions = [{'tool': 'execute_bash', 'params': {'command': 'echo ok'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'success'
        agent.tools.execute_bash.assert_called_once_with('echo ok')

    def test_max_per_cycle_cap(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.max_immediate_per_cycle = 2
        agent.tools = MagicMock()
        agent.tools.read_file.return_value = ToolResult(success=True, output="ok")

        actions = [
            {'tool': 'read_file', 'params': {'path': f'{i}.md'}, 'immediate': True}
            for i in range(5)
        ]
        results = agent._execute_immediate_actions(actions)

        executed = [r for r in results if r['status'] == 'success']
        skipped = [r for r in results if r['status'] == 'skipped_max_reached']
        assert len(executed) == 2
        assert len(skipped) == 3

    def test_deferred_creates_decision(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.cycle_number = 10

        actions = [{'tool': 'delete_file', 'params': {'path': 'foo.md'}, 'immediate': False}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'deferred'
        agent.decisions.create_decision.assert_called_once()

    def test_error_result_captured(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.read_file.return_value = ToolResult(
            success=False, output=None, error="File not found"
        )

        actions = [{'tool': 'read_file', 'params': {'path': 'nope.md'}, 'immediate': True}]
        results = agent._execute_immediate_actions(actions)

        assert results[0]['status'] == 'error'
        assert results[0]['error'] == 'File not found'

    def test_disabled_flag(self, temp_dir, mock_config):
        """When immediate_actions.enabled is False, no actions are parsed/executed."""
        mock_config['immediate_actions']['enabled'] = False
        agent = _make_agent(temp_dir, mock_config)
        assert agent.immediate_actions_enabled is False


# ===========================================================================
# Feedback loop
# ===========================================================================

class TestActionResultsFeedback:
    """Test that action results persist to logs and feed into next cycle."""

    def test_results_in_cycle_log(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.cycle_number = 5

        from agent import CycleState
        cs = CycleState(cycle_number=5, start_time="2025-01-01T00:00:00Z")
        cs.action_results = [
            {'tool': 'read_file', 'params': {'path': 'x'}, 'status': 'success',
             'output': 'contents', 'error': None}
        ]

        # Mock out components that _save_state calls
        agent.decisions = MagicMock()
        agent.goals = MagicMock()

        agent._save_state(cs)

        log_file = temp_dir / "logs" / "cycle_00005.json"
        assert log_file.exists()
        data = json.loads(log_file.read_text())
        assert data['immediate_actions_executed'] == 1
        assert len(data['action_results']) == 1

    def test_previous_results_loaded(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.cycle_number = 6

        # Write a fake previous cycle log
        log_dir = temp_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        prev_log = log_dir / "cycle_00005.json"
        prev_log.write_text(json.dumps({
            'cycle_number': 5,
            'action_results': [
                {'tool': 'write_file', 'status': 'success', 'output': 'Written'}
            ]
        }))

        results = agent._get_previous_action_results()
        assert len(results) == 1
        assert results[0]['tool'] == 'write_file'

    def test_no_previous_log(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.cycle_number = 1
        assert agent._get_previous_action_results() == []


# ===========================================================================
# Escape handling
# ===========================================================================

class TestEscapeHandling:
    """Test that escaped characters in params are handled correctly."""

    def test_newline_escape_in_write_file(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.write_file.return_value = ToolResult(success=True, output="Written")

        actions = [{'tool': 'write_file',
                     'params': {'path': 'test.md', 'content': 'line1\\nline2'},
                     'immediate': True}]
        agent._execute_immediate_actions(actions)

        agent.tools.write_file.assert_called_once_with('test.md', 'line1\nline2')

    def test_newline_escape_in_execute_python(self, temp_dir, mock_config):
        agent = _make_agent(temp_dir, mock_config)
        agent.tools = MagicMock()
        agent.tools.execute_python.return_value = ToolResult(success=True, output="ok")

        actions = [{'tool': 'execute_python',
                     'params': {'code': 'print("a")\\nprint("b")'},
                     'immediate': True}]
        agent._execute_immediate_actions(actions)

        agent.tools.execute_python.assert_called_once_with('print("a")\nprint("b")')
