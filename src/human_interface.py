"""
Human Interface for Coeus

Manages file-based communication between Coeus and human observers.
Handles pending decisions, human responses, and conversation logging.
"""

import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class HumanMessage:
    """A message from the human observer."""
    timestamp: str
    content: str
    decision_id: Optional[str] = None
    response_type: Optional[str] = None  # APPROVED, DENIED, NEEDS_MORE_INFO, GENERAL


class HumanInterface:
    """
    Manages bidirectional file-based communication with humans.
    
    Files:
    - pending_decisions.md: Coeus writes decisions needing approval
    - human_responses.md: Human writes responses
    - conversation_log.md: Running log of all interactions
    - general_input.md: Human can write general messages/guidance
    """
    
    def __init__(self, interaction_path: str):
        self.path = Path(interaction_path)
        self.path.mkdir(parents=True, exist_ok=True)
        
        self.pending_file = self.path / "pending_decisions.md"
        self.response_file = self.path / "human_responses.md"
        self.log_file = self.path / "conversation_log.md"
        self.input_file = self.path / "general_input.md"
        
        self._initialize_files()
        
        # Track what we've already processed
        self._processed_responses = set()
        self._processed_inputs = set()
    
    def _initialize_files(self):
        """Create initial files if they don't exist."""
        if not self.pending_file.exists():
            self.pending_file.write_text("""# Pending Decisions

Coeus will write decisions here that require human approval.
These are "one-way door" decisions that are irreversible or high-impact.

---

""")
        
        if not self.response_file.exists():
            self.response_file.write_text("""# Human Responses

Write your responses to Coeus's decisions here.

Format:
```
## Response to [decision-id]
**Decision**: APPROVED | DENIED | NEEDS_MORE_INFO
**Notes**: Your optional feedback here
```

---

""")
        
        if not self.log_file.exists():
            self.log_file.write_text(f"""# Conversation Log

Started: {datetime.now(timezone.utc).isoformat()}

---

""")
        
        if not self.input_file.exists():
            self.input_file.write_text("""# General Input to Coeus

Write messages here that Coeus will read each cycle.
You can provide guidance, ask questions, or share observations.

After Coeus reads a message, it will be logged and you can delete it
or leave it for reference.

---

""")
    
    def log_message(self, source: str, message: str):
        """Log a message to the conversation log."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"[{timestamp}] **{source}**: {message}\n\n"
        
        with open(self.log_file, 'a') as f:
            f.write(entry)
    
    def write_pending_decision(
        self,
        decision_id: str,
        decision_type: str,
        summary: str,
        reasoning: str,
        counterarguments: list[str],
        confidence: float,
        conviction_cycles: int,
        required_cycles: int,
        created_cycle: int
    ):
        """Write a decision to the pending file for human review."""
        entry = f"""
## Decision: {decision_id}

**Type**: {decision_type.upper()}
**Status**: AWAITING HUMAN REVIEW
**Created**: Cycle {created_cycle}
**Timestamp**: {datetime.now(timezone.utc).isoformat()}

### Summary
{summary}

### Reasoning
{reasoning}

### Counterarguments Considered
{chr(10).join('- ' + ca for ca in counterarguments) if counterarguments else '- None documented'}

### Confidence & Conviction
- Confidence: {confidence * 100:.1f}%
- Conviction cycles: {conviction_cycles}/{required_cycles}

### How to Respond
Add a response in `human_responses.md` with this format:
```
## Response to {decision_id}
**Decision**: APPROVED | DENIED | NEEDS_MORE_INFO
**Notes**: (optional) Your feedback
```

---

"""
        with open(self.pending_file, 'a') as f:
            f.write(entry)
        
        self.log_message("COEUS", f"Decision {decision_id} submitted for review: {summary}")
    
    def check_responses(self) -> list[tuple[str, str, str]]:
        """
        Check for new human responses.
        
        Returns list of (decision_id, response, notes) that haven't been processed.
        """
        try:
            content = self.response_file.read_text()
        except FileNotFoundError:
            return []
        
        # Parse responses
        pattern = r'## Response to (\S+)\s*\n\*\*Decision\*\*:\s*(APPROVED|DENIED|NEEDS_MORE_INFO)\s*(?:\n\*\*Notes\*\*:\s*(.+?))?(?=\n## |\n---|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        new_responses = []
        for decision_id, response, notes in matches:
            decision_id = decision_id.strip()
            # Create a hash to track processed responses
            response_hash = f"{decision_id}:{response}:{notes[:50] if notes else ''}"
            
            if response_hash not in self._processed_responses:
                self._processed_responses.add(response_hash)
                new_responses.append((decision_id, response.strip(), notes.strip() if notes else ""))
                self.log_message("HUMAN", f"Response to {decision_id}: {response}")
        
        return new_responses
    
    def check_general_input(self) -> list[str]:
        """
        Check for new general input from the human.
        
        Returns list of new messages.
        """
        try:
            content = self.input_file.read_text()
        except FileNotFoundError:
            return []
        
        # Find messages (lines that aren't headers or empty)
        messages = []
        in_message = False
        current_message = []
        
        for line in content.split('\n'):
            # Skip header lines and separators
            if line.startswith('#') or line.startswith('---') or not line.strip():
                if current_message:
                    msg = '\n'.join(current_message).strip()
                    if msg and msg not in self._processed_inputs:
                        messages.append(msg)
                        self._processed_inputs.add(msg)
                    current_message = []
                continue
            
            # Skip the instruction text
            if 'Write messages here' in line or 'After Coeus reads' in line:
                continue
            
            current_message.append(line)
        
        # Don't forget last message
        if current_message:
            msg = '\n'.join(current_message).strip()
            if msg and msg not in self._processed_inputs:
                messages.append(msg)
                self._processed_inputs.add(msg)
        
        for msg in messages:
            self.log_message("HUMAN", f"General input: {msg[:100]}...")
        
        return messages
    
    def write_to_human(self, message: str, category: str = "UPDATE"):
        """
        Write a message to the human.
        
        This goes in the conversation log and could trigger notifications.
        """
        self.log_message(f"COEUS ({category})", message)
    
    def mark_decision_resolved(self, decision_id: str, resolution: str):
        """
        Mark a decision as resolved in the pending file.
        
        This updates the status so humans know it's been handled.
        """
        try:
            content = self.pending_file.read_text()
            
            # Find and update the decision entry
            pattern = rf'(## Decision: {re.escape(decision_id)}.*?\*\*Status\*\*: )AWAITING HUMAN REVIEW'
            replacement = rf'\1RESOLVED ({resolution})'
            
            updated = re.sub(pattern, replacement, content, flags=re.DOTALL)
            self.pending_file.write_text(updated)
        except Exception as e:
            self.log_message("SYSTEM", f"Error updating pending decision: {e}")
    
    def get_interaction_summary(self) -> str:
        """Get a summary of recent interactions for the agent's context."""
        summary_parts = []
        
        # Check for pending human responses needed
        pending_content = self.pending_file.read_text() if self.pending_file.exists() else ""
        awaiting_count = pending_content.count("AWAITING HUMAN REVIEW")
        if awaiting_count > 0:
            summary_parts.append(f"⏳ {awaiting_count} decision(s) awaiting human review")
        
        # Check for unread general input
        general_input = self.check_general_input()
        if general_input:
            summary_parts.append(f"📬 {len(general_input)} new message(s) from human")
        
        # Recent log entries
        try:
            log_content = self.log_file.read_text()
            lines = log_content.strip().split('\n')
            recent = [l for l in lines[-10:] if l.strip() and not l.startswith('#')]
            if recent:
                summary_parts.append("Recent interactions:\n" + "\n".join(recent[-5:]))
        except FileNotFoundError:
            pass
        
        return "\n".join(summary_parts) if summary_parts else "No recent human interactions"


def format_human_input_for_prompt(interface: HumanInterface) -> str:
    """Format human input for inclusion in the agent's prompt."""
    sections = []
    
    # Check for general input
    messages = interface.check_general_input()
    if messages:
        sections.append("## Messages from Human Observer")
        for msg in messages:
            sections.append(f"> {msg}")
    
    # Check for decision responses
    responses = interface.check_responses()
    if responses:
        sections.append("\n## Decision Responses")
        for decision_id, response, notes in responses:
            sections.append(f"- **{decision_id}**: {response}")
            if notes:
                sections.append(f"  Notes: {notes}")
    
    return "\n".join(sections) if sections else ""
