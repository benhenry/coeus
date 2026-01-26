"""
Tests for the tools and capability management module.
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools import (
    CapabilityManager, ToolStatus, SandboxedTools, ToolResult,
    WebSearchTool, WebFetchTool,
    format_tool_result, get_available_tools_description
)


class TestCapabilityManager:
    """Tests for the CapabilityManager class."""

    def test_default_capabilities(self, temp_dir, mock_config):
        """Test that default capabilities are loaded from config."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        caps = manager.list_capabilities()
        assert 'code_execution' in caps
        assert caps['code_execution'] == 'enabled'
        assert 'web_search' in caps
        assert caps['web_search'] == 'disabled'

    def test_is_enabled(self, temp_dir, mock_config):
        """Test checking if a capability is enabled."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        assert manager.is_enabled('code_execution') is True
        assert manager.is_enabled('web_search') is False
        # Nonexistent capabilities return falsy (None or False)
        assert not manager.is_enabled('nonexistent')

    def test_enable_capability(self, temp_dir, mock_config):
        """Test enabling a capability."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        manager.enable('web_search')
        assert manager.is_enabled('web_search') is True

    def test_disable_capability(self, temp_dir, mock_config):
        """Test disabling a capability."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        manager.disable('code_execution')
        assert manager.is_enabled('code_execution') is False
        caps = manager.list_capabilities()
        assert caps['code_execution'] == 'disabled'

    def test_state_persistence(self, temp_dir, mock_config):
        """Test that capability state persists."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        manager.enable('web_search')

        # Load into new manager
        new_manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        assert new_manager.is_enabled('web_search') is True

    def test_get_config(self, temp_dir, mock_config):
        """Test getting capability configuration."""
        state_path = temp_dir / "capabilities.json"
        manager = CapabilityManager(
            config=mock_config,
            state_path=str(state_path)
        )

        config = manager.get_config('code_execution')
        assert config.get('enabled') is True


class TestSandboxedTools:
    """Tests for the SandboxedTools class."""

    def test_workspace_creation(self, temp_dir, mock_config):
        """Test that workspace directory is created."""
        workspace_path = temp_dir / "workspace"
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        assert workspace_path.exists()

    def test_write_file_in_workspace(self, temp_dir, mock_config):
        """Test writing a file within the workspace."""
        workspace_path = temp_dir / "workspace"
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.write_file("test.txt", "Hello, World!")

        assert result.success is True
        assert (workspace_path / "test.txt").exists()
        assert (workspace_path / "test.txt").read_text() == "Hello, World!"

    def test_read_file_in_workspace(self, temp_dir, mock_config):
        """Test reading a file within the workspace."""
        workspace_path = temp_dir / "workspace"
        workspace_path.mkdir(parents=True)
        (workspace_path / "existing.txt").write_text("Existing content")

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.read_file("existing.txt")

        assert result.success is True
        assert result.output == "Existing content"

    def test_cannot_read_outside_workspace(self, temp_dir, mock_config):
        """Test that files outside workspace cannot be read."""
        workspace_path = temp_dir / "workspace"
        outside_file = temp_dir / "outside.txt"
        outside_file.write_text("Secret content")

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.read_file("../outside.txt")

        assert result.success is False
        assert result.error is not None

    def test_cannot_write_outside_workspace(self, temp_dir, mock_config):
        """Test that files outside workspace cannot be written."""
        workspace_path = temp_dir / "workspace"

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.write_file("../outside.txt", "Malicious content")

        assert result.success is False
        assert not (temp_dir / "outside.txt").exists()

    def test_execute_python_simple(self, temp_dir, mock_config):
        """Test executing simple Python code."""
        workspace_path = temp_dir / "workspace"

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.execute_python("print(2 + 2)")

        assert result.success is True
        assert "4" in result.output

    def test_execute_python_with_error(self, temp_dir, mock_config):
        """Test executing Python code that raises an error."""
        workspace_path = temp_dir / "workspace"

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.execute_python("raise ValueError('test error')")

        assert result.success is False
        assert "ValueError" in result.error or "error" in str(result).lower()

    def test_get_workspace_state(self, temp_dir, mock_config):
        """Test getting workspace state."""
        workspace_path = temp_dir / "workspace"
        workspace_path.mkdir(parents=True)
        (workspace_path / "file1.txt").write_text("Content 1")
        (workspace_path / "file2.py").write_text("print('hello')")
        (workspace_path / "subdir").mkdir()

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        state = tools.get_workspace_state()

        assert state['file_count'] >= 2
        assert 'file1.txt' in state['top_level_items']
        assert 'subdir' in state['top_level_items']

    def test_code_execution_disabled(self, temp_dir, mock_config):
        """Test that code execution can be disabled."""
        mock_config['tools']['code_execution']['enabled'] = False
        workspace_path = temp_dir / "workspace"

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        tools = SandboxedTools(
            workspace_path=str(workspace_path),
            capability_manager=cap_manager
        )

        result = tools.execute_python("print('hello')")

        assert result.success is False
        assert 'not enabled' in result.error.lower()


class TestFormatToolResult:
    """Tests for the format_tool_result function."""

    def test_format_success_result(self):
        """Test formatting a successful result."""
        result = ToolResult(
            success=True,
            output='Hello, World!'
        )

        formatted = format_tool_result(result, 'python')

        assert 'Hello, World!' in formatted
        assert 'Success' in formatted

    def test_format_error_result(self):
        """Test formatting an error result."""
        result = ToolResult(
            success=False,
            output=None,
            error='Something went wrong'
        )

        formatted = format_tool_result(result, 'python')

        assert 'Failed' in formatted or 'error' in formatted.lower()
        assert 'Something went wrong' in formatted


class TestGetAvailableToolsDescription:
    """Tests for get_available_tools_description function."""

    def test_describes_enabled_tools(self, temp_dir, mock_config):
        """Test that enabled tools are described."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        description = get_available_tools_description(cap_manager)

        assert 'code_execution' in description.lower() or 'Code' in description
        assert 'enabled' in description.lower()

    def test_describes_disabled_tools(self, temp_dir, mock_config):
        """Test that disabled tools are mentioned."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        description = get_available_tools_description(cap_manager)

        assert 'web_search' in description.lower() or 'search' in description.lower()
        assert 'disabled' in description.lower()


class TestWebSearchTool:
    """Tests for the WebSearchTool class."""

    def test_search_disabled_by_default(self, temp_dir, mock_config):
        """Test that search fails when capability is disabled."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        search_tool = WebSearchTool(cap_manager)
        result = search_tool.search("test query")

        assert result.success is False
        assert "not enabled" in result.error.lower()

    def test_search_requires_api_key(self, temp_dir, mock_config):
        """Test that search fails without API key."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )
        cap_manager.enable('web_search')

        # Create tool without API key
        search_tool = WebSearchTool(cap_manager, api_key=None)
        # Clear any env var
        import os
        old_key = os.environ.pop('TAVILY_API_KEY', None)

        try:
            result = search_tool.search("test query")
            assert result.success is False
            assert "api_key" in result.error.lower() or "tavily" in result.error.lower()
        finally:
            if old_key:
                os.environ['TAVILY_API_KEY'] = old_key

    def test_get_search_context_disabled(self, temp_dir, mock_config):
        """Test that get_search_context fails when disabled."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        search_tool = WebSearchTool(cap_manager)
        result = search_tool.get_search_context("test query")

        assert result.success is False
        assert "not enabled" in result.error.lower()


class TestWebFetchTool:
    """Tests for the WebFetchTool class."""

    def test_fetch_disabled_by_default(self, temp_dir, mock_config):
        """Test that fetch fails when capability is disabled."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )

        fetch_tool = WebFetchTool(cap_manager)
        result = fetch_tool.fetch("https://example.com")

        assert result.success is False
        assert "not enabled" in result.error.lower()

    def test_fetch_rejects_invalid_url(self, temp_dir, mock_config):
        """Test that fetch rejects invalid URLs."""
        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )
        cap_manager.enable('web_fetch')

        fetch_tool = WebFetchTool(cap_manager)
        result = fetch_tool.fetch("not-a-valid-url")

        assert result.success is False
        assert "invalid url" in result.error.lower()

    def test_fetch_enabled_makes_request(self, temp_dir, mock_config):
        """Test that fetch works when enabled (mocked request)."""
        from unittest.mock import patch, MagicMock

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )
        cap_manager.enable('web_fetch')

        fetch_tool = WebFetchTool(cap_manager, timeout=5)

        # Mock the requests.get call
        mock_response = MagicMock()
        mock_response.text = "<html><head><title>Test Page</title></head><body><p>Hello World</p></body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.status_code = 200

        with patch('requests.get', return_value=mock_response) as mock_get:
            result = fetch_tool.fetch("https://example.com")

            mock_get.assert_called_once()
            assert result.success is True
            assert result.output['url'] == "https://example.com"
            assert result.output['title'] == "Test Page"
            assert "Hello World" in result.output['content']

    def test_fetch_handles_timeout(self, temp_dir, mock_config):
        """Test that fetch handles timeout gracefully."""
        from unittest.mock import patch
        import requests

        cap_manager = CapabilityManager(
            config=mock_config,
            state_path=str(temp_dir / "caps.json")
        )
        cap_manager.enable('web_fetch')

        fetch_tool = WebFetchTool(cap_manager, timeout=5)

        with patch('requests.get', side_effect=requests.exceptions.Timeout()):
            result = fetch_tool.fetch("https://example.com")

            assert result.success is False
            assert "timed out" in result.error.lower()
