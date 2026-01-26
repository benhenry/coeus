"""
Sandboxed Tools for Coeus

Provides code execution and file system access within the sandbox,
with capability management for tools that require approval.
"""

import os
import sys
import json
import subprocess
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
from pathlib import Path
import time


class ToolStatus(Enum):
    """Status of a tool/capability."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PENDING_APPROVAL = "pending_approval"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0
    

class CapabilityManager:
    """
    Manages which tools/capabilities Coeus has access to.
    
    Some capabilities are enabled by default, others must be
    requested through the decision framework.
    """
    
    def __init__(self, config: dict, state_path: str):
        self.state_path = Path(state_path)
        self.capabilities = self._load_state(config)
    
    def _load_state(self, config: dict) -> dict:
        """Load capability state, merging with config defaults."""
        # Start with config defaults
        caps = {}
        for name, tool_config in config.get('tools', {}).items():
            caps[name] = {
                'status': ToolStatus.ENABLED if tool_config.get('enabled') else ToolStatus.DISABLED,
                'config': tool_config
            }
        
        # Override with saved state
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            for name, state in saved.items():
                if name in caps:
                    caps[name]['status'] = ToolStatus(state['status'])
        
        return caps
    
    def save_state(self):
        """Save capability state."""
        data = {
            name: {'status': cap['status'].value}
            for name, cap in self.capabilities.items()
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(data, indent=2))
    
    def is_enabled(self, capability: str) -> bool:
        """Check if a capability is enabled."""
        cap = self.capabilities.get(capability)
        return cap and cap['status'] == ToolStatus.ENABLED
    
    def enable(self, capability: str):
        """Enable a capability (after approval)."""
        if capability in self.capabilities:
            self.capabilities[capability]['status'] = ToolStatus.ENABLED
            self.save_state()
    
    def disable(self, capability: str):
        """Disable a capability."""
        if capability in self.capabilities:
            self.capabilities[capability]['status'] = ToolStatus.DISABLED
            self.save_state()
    
    def get_config(self, capability: str) -> dict:
        """Get configuration for a capability."""
        cap = self.capabilities.get(capability)
        return cap['config'] if cap else {}
    
    def list_capabilities(self) -> dict[str, str]:
        """List all capabilities and their status."""
        return {
            name: cap['status'].value
            for name, cap in self.capabilities.items()
        }


class SandboxedTools:
    """
    Provides sandboxed tool execution for Coeus.
    
    All file operations are restricted to the workspace directory.
    Code execution happens in isolated subprocesses.
    """
    
    def __init__(self, workspace_path: str, capability_manager: CapabilityManager):
        self.workspace = Path(workspace_path).resolve()
        self.capabilities = capability_manager
        
        # Ensure workspace exists
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate that a path is within the sandbox.
        
        Raises ValueError if path escapes sandbox.
        """
        # Resolve to absolute path
        if not os.path.isabs(path):
            full_path = (self.workspace / path).resolve()
        else:
            full_path = Path(path).resolve()
        
        # Check it's within workspace
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path {path} is outside sandbox")
        
        return full_path
    
    def execute_python(self, code: str, timeout: int = 30) -> ToolResult:
        """
        Execute Python code in a subprocess.
        
        Returns stdout/stderr and success status.
        """
        if not self.capabilities.is_enabled('code_execution'):
            return ToolResult(
                success=False,
                output=None,
                error="Code execution capability not enabled"
            )
        
        start = time.time()
        
        # Write code to temp file in workspace
        code_file = self.workspace / "_temp_code.py"
        code_file.write_text(code)
        
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace)
            )
            
            execution_time = (time.time() - start) * 1000
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output=result.stdout,
                    execution_time_ms=execution_time
                )
            else:
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time_ms=execution_time
                )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
        finally:
            # Clean up temp file
            if code_file.exists():
                code_file.unlink()
    
    def execute_bash(self, command: str, timeout: int = 30) -> ToolResult:
        """
        Execute a bash command in the sandbox.
        """
        if not self.capabilities.is_enabled('code_execution'):
            return ToolResult(
                success=False,
                output=None,
                error="Code execution capability not enabled"
            )
        
        start = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace)
            )
            
            execution_time = (time.time() - start) * 1000
            
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_ms=execution_time
            )
        
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def read_file(self, path: str) -> ToolResult:
        """Read a file from the workspace."""
        if not self.capabilities.is_enabled('file_system'):
            return ToolResult(
                success=False,
                output=None,
                error="File system capability not enabled"
            )
        
        try:
            full_path = self._validate_path(path)
            content = full_path.read_text()
            return ToolResult(success=True, output=content)
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except FileNotFoundError:
            return ToolResult(success=False, output=None, error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file in the workspace."""
        if not self.capabilities.is_enabled('file_system'):
            return ToolResult(
                success=False,
                output=None,
                error="File system capability not enabled"
            )
        
        try:
            full_path = self._validate_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return ToolResult(success=True, output=f"Written to {path}")
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def list_directory(self, path: str = ".") -> ToolResult:
        """List contents of a directory in the workspace."""
        if not self.capabilities.is_enabled('file_system'):
            return ToolResult(
                success=False,
                output=None,
                error="File system capability not enabled"
            )
        
        try:
            full_path = self._validate_path(path)
            if not full_path.is_dir():
                return ToolResult(success=False, output=None, error=f"Not a directory: {path}")
            
            contents = []
            for item in full_path.iterdir():
                rel_path = item.relative_to(self.workspace)
                item_type = "dir" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else 0
                contents.append({
                    'name': item.name,
                    'path': str(rel_path),
                    'type': item_type,
                    'size': size
                })
            
            return ToolResult(success=True, output=contents)
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def delete_file(self, path: str) -> ToolResult:
        """
        Delete a file from the workspace.
        
        Note: This is a potentially dangerous operation and might
        be flagged as a one-way door depending on context.
        """
        if not self.capabilities.is_enabled('file_system'):
            return ToolResult(
                success=False,
                output=None,
                error="File system capability not enabled"
            )
        
        try:
            full_path = self._validate_path(path)
            if full_path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error="Use delete_directory for directories"
                )
            full_path.unlink()
            return ToolResult(success=True, output=f"Deleted {path}")
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))
        except FileNotFoundError:
            return ToolResult(success=False, output=None, error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def get_workspace_state(self) -> dict:
        """
        Get a summary of the workspace state.
        
        Used for environmental context capture.
        """
        try:
            file_count = sum(1 for _ in self.workspace.rglob('*') if _.is_file())
            dir_count = sum(1 for _ in self.workspace.rglob('*') if _.is_dir())
            total_size = sum(f.stat().st_size for f in self.workspace.rglob('*') if f.is_file())
            
            # List top-level items
            top_level = [
                item.name for item in self.workspace.iterdir()
                if not item.name.startswith('_')  # Skip temp files
            ]
            
            return {
                'file_count': file_count,
                'directory_count': dir_count,
                'total_size_bytes': total_size,
                'top_level_items': top_level
            }
        except Exception as e:
            return {'error': str(e)}


class WebSearchTool:
    """
    Web search capability using Tavily API.

    Requires approval to enable. Uses Tavily for AI-optimized search results.
    """

    def __init__(self, capability_manager: CapabilityManager, api_key: Optional[str] = None):
        self.capabilities = capability_manager
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazily initialize the Tavily client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("TAVILY_API_KEY not set")
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("tavily-python not installed. Run: pip install tavily-python")
        return self._client

    def search(
        self,
        query: str,
        num_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None
    ) -> ToolResult:
        """
        Search the web for information.

        Args:
            query: The search query
            num_results: Number of results to return (max 10)
            search_depth: "basic" or "advanced" (advanced is slower but more thorough)
            include_domains: Only search these domains
            exclude_domains: Exclude these domains from results

        Returns:
            ToolResult with search results or error
        """
        if not self.capabilities.is_enabled('web_search'):
            return ToolResult(
                success=False,
                output=None,
                error="Web search capability not enabled. Request this capability through the decision framework."
            )

        if not self.api_key:
            return ToolResult(
                success=False,
                output=None,
                error="TAVILY_API_KEY environment variable not set."
            )

        start = time.time()

        try:
            client = self._get_client()

            # Build search options
            search_opts = {
                "query": query,
                "search_depth": search_depth,
                "max_results": min(num_results, 10)
            }
            if include_domains:
                search_opts["include_domains"] = include_domains
            if exclude_domains:
                search_opts["exclude_domains"] = exclude_domains

            response = client.search(**search_opts)

            # Format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })

            execution_time = (time.time() - start) * 1000

            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": results,
                    "answer": response.get("answer"),  # Tavily can provide a direct answer
                    "result_count": len(results)
                },
                execution_time_ms=execution_time
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Search failed: {str(e)}"
            )

    def get_search_context(self, query: str, max_tokens: int = 4000) -> ToolResult:
        """
        Get a condensed context from search results, optimized for LLM consumption.

        This uses Tavily's context feature which returns a single string
        suitable for feeding into an LLM.
        """
        if not self.capabilities.is_enabled('web_search'):
            return ToolResult(
                success=False,
                output=None,
                error="Web search capability not enabled."
            )

        if not self.api_key:
            return ToolResult(
                success=False,
                output=None,
                error="TAVILY_API_KEY environment variable not set."
            )

        try:
            client = self._get_client()
            context = client.get_search_context(query=query, max_tokens=max_tokens)

            return ToolResult(
                success=True,
                output={"context": context, "query": query}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Context retrieval failed: {str(e)}"
            )


class WebFetchTool:
    """
    Web page fetching capability.

    Fetches and extracts text content from web pages.
    Requires approval to enable.
    """

    def __init__(self, capability_manager: CapabilityManager, timeout: int = 30):
        self.capabilities = capability_manager
        self.timeout = timeout

    def fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_length: int = 10000
    ) -> ToolResult:
        """
        Fetch content from a URL.

        Args:
            url: The URL to fetch
            extract_text: If True, extract and clean text content; if False, return raw HTML
            max_length: Maximum content length to return

        Returns:
            ToolResult with page content or error
        """
        if not self.capabilities.is_enabled('web_fetch'):
            return ToolResult(
                success=False,
                output=None,
                error="Web fetch capability not enabled. Request this capability through the decision framework."
            )

        start = time.time()

        try:
            import requests
            from bs4 import BeautifulSoup

            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return ToolResult(
                    success=False,
                    output=None,
                    error="Invalid URL: must start with http:// or https://"
                )

            # Fetch the page
            headers = {
                'User-Agent': 'Coeus-Agent/1.0 (Autonomous AI Assistant)'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '')

            if extract_text and 'text/html' in content_type:
                # Parse HTML and extract text
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove script and style elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                # Get text content
                text = soup.get_text(separator='\n', strip=True)

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)

                # Truncate if needed
                if len(text) > max_length:
                    text = text[:max_length] + "\n... [truncated]"

                content = text
                title = soup.title.string if soup.title else None
            else:
                # Return raw content (or portion of it)
                content = response.text[:max_length]
                if len(response.text) > max_length:
                    content += "\n... [truncated]"
                title = None

            execution_time = (time.time() - start) * 1000

            return ToolResult(
                success=True,
                output={
                    "url": url,
                    "title": title,
                    "content": content,
                    "content_type": content_type,
                    "status_code": response.status_code,
                    "content_length": len(content)
                },
                execution_time_ms=execution_time
            )

        except ImportError:
            return ToolResult(
                success=False,
                output=None,
                error="requests or beautifulsoup4 not installed. Run: pip install requests beautifulsoup4"
            )
        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request timed out after {self.timeout} seconds"
            )
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Request failed: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Fetch failed: {str(e)}"
            )


def format_tool_result(result: ToolResult, tool_name: str) -> str:
    """Format a tool result for inclusion in agent context."""
    if result.success:
        output_str = str(result.output)
        if len(output_str) > 1000:
            output_str = output_str[:1000] + "... (truncated)"
        return f"[{tool_name}] Success:\n{output_str}"
    else:
        return f"[{tool_name}] Failed: {result.error}"


def get_available_tools_description(capability_manager: CapabilityManager) -> str:
    """Get a description of available tools for the agent."""
    caps = capability_manager.list_capabilities()
    
    descriptions = []
    for name, status in caps.items():
        status_icon = "✓" if status == "enabled" else "✗"
        descriptions.append(f"- {status_icon} {name}: {status}")
    
    return "## Available Tools\n" + "\n".join(descriptions)
