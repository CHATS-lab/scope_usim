"""CooperBench agent implementing BaseAgent protocol.

Renders system/instance prompts from CooperBench mini-swe-agent-v2 templates
(config/mini.yaml) for coding agents that use OpenAI-style tool calling.

The templates here are a direct Python translation of the Jinja2 templates in:
  external/CooperBench/src/cooperbench/agents/mini_swe_agent_v2/config/mini.yaml
"""

import logging
import platform
from typing import Any, Dict, List, Optional, Tuple

from usim.core.types import AgentState, Message

logger = logging.getLogger(__name__)

STOP_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


# ---------------------------------------------------------------------------
# Shared suffix: Command Execution Rules + Example + System Info + Commands
# This appears at the end of ALL instance templates (both collab and non-collab).
# ---------------------------------------------------------------------------

_COMMAND_RULES_EXAMPLE_AND_COMMANDS = """\

## Command Execution Rules

You are operating in an environment where you issue bash commands via tool calls.

**CRITICAL REQUIREMENTS:**

- Every response MUST include at least one bash tool call
- Directory/env changes are NOT persistent — prefix with `cd /path && ...`
- Submit when done: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (alone, no other commands)

<important>You have very limited context. Do NOT spend more than 1 step exploring. Read the task description carefully — it tells you what file to edit. Start writing code in your SECOND response at the latest.</important>

## Example: Complete Coding Workflow

### Step 1 — Quick look at the target file:
<example_response>
The task says to modify `src/utils.py`. Let me check its current contents.

[Makes bash tool call with {"command": "cat src/utils.py"} as arguments]
</example_response>

### Step 2 — Write the code (use sed or cat to edit/create files):
<example_response>
I can see the existing code. I'll add the new function after line 15.

[Makes bash tool call with {"command": "sed -i '15a\\\\ndef new_feature():\\n    return 42' src/utils.py"} as arguments]
</example_response>

### Step 3 — Submit:
<example_response>
The changes are implemented. Time to submit.

[Makes bash tool call with {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"} as arguments]
</example_response>

<system_information>
%s
</system_information>

## Useful commands

```bash
# Create a new file
cat <<'EOF' > newfile.py
import numpy as np
print("hello")
EOF

# Edit with sed
sed -i 's/old/new/g' file.py          # Replace all
sed -i '10,20s/old/new/g' file.py     # Lines 10-20 only

# View file
nl -ba file.py | sed -n '10,20p'
```"""


def _system_info_string() -> str:
    return f"{platform.system()} {platform.release()} {platform.version()} {platform.machine()}"


def _shared_suffix() -> str:
    return _COMMAND_RULES_EXAMPLE_AND_COMMANDS % _system_info_string()


# ---------------------------------------------------------------------------
# Prompt rendering (matches mini.yaml system_template / instance_template)
# ---------------------------------------------------------------------------


def _render_system_prompt(
    agent_id: Optional[str] = None,
    agents: Optional[List[str]] = None,
    messaging_enabled: bool = False,
) -> str:
    """Render system prompt matching mini.yaml system_template."""
    prompt = "You are an expert software engineer. You write code quickly and efficiently. When given a coding task, you read the target file once and then immediately implement the changes.\n"
    if agent_id and agents and len(agents) > 1:
        agents_str = ", ".join(agents)
        prompt += f"\nYou are {agent_id} working as a team with: {agents_str}.\n"
        prompt += "You are all working on related features in the same codebase. Each agent has their own workspace.\n"
        if messaging_enabled:
            prompt += "\nUse the send_message tool to coordinate with teammates.\n"
    return prompt


def _render_instance_template(
    task: str,
    agent_id: Optional[str] = None,
    agents: Optional[List[str]] = None,
    messaging_enabled: bool = False,
) -> str:
    """Render instance template matching mini.yaml instance_template."""
    is_collab = bool(agent_id and agents and len(agents) > 1)

    text = f"Please solve this issue: {task}\n"
    text += "\nYou can execute bash commands and edit files to implement the necessary changes.\n"

    if is_collab:
        # --- Collaboration section ---
        text += "\n## Collaboration\n\n"
        text += "Each agent has their own workspace. At the end, all agents' changes will be merged together.\n"
        text += "**Important**: Coordinate to avoid merge conflicts - your patches must cleanly combine!\n"

        if messaging_enabled:
            text += "\n### Messaging\n"
            text += "Use the `send_message` tool with `recipient` and `content` arguments to coordinate with teammates.\n"
            text += "Messages from teammates appear as: [Message from <agent_name>]: ...\n"

        # --- Collab workflow ---
        text += "\n## Workflow\n\n"
        text += "1. Explore the codebase and find relevant files for your feature\n"
        if messaging_enabled:
            text += "2. Message teammates about which files you plan to modify\n"
        text += "3. Implement your changes\n"
        text += "4. Test your changes work\n"
        text += "5. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
    else:
        # --- Non-collab (baseline/solo) workflow ---
        text += "\n## Workflow\n\n"
        text += "<important>You have VERY limited context (~5 steps). You MUST start writing code by step 2.</important>\n\n"
        text += "1. Read the target file mentioned in the task (1 command)\n"
        text += "2. Write/edit the code using `sed -i` or `cat <<'EOF' > file`\n"
        text += "3. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
        text += "\nDo NOT explore the repo structure. The task description tells you which files to modify. Go directly to those files.\n"

    # --- Shared suffix (command rules, example, system info, useful commands) ---
    text += _shared_suffix()

    return text


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class CooperBenchAgent:
    """Agent for CooperBench coding tasks using mini-swe-agent-v2 tool calling.

    Renders system and instance prompts from CooperBench's v2 templates,
    with support for multi-agent collaboration via the send_message tool.
    """

    def __init__(
        self,
        agent_id: str = "agent1",
        agents: Optional[List[str]] = None,
        messaging_enabled: bool = True,
        setting: str = "coop",
    ):
        self.agent_id = agent_id
        self.setting = setting
        # For baseline/solo, disable collaboration (single agent)
        if setting in ("baseline", "solo"):
            self.agents = [agent_id]
            self.messaging_enabled = False
        else:
            self.agents = agents or [agent_id]
            self.messaging_enabled = messaging_enabled

        is_collab = len(self.agents) > 1
        self._system_prompt = _render_system_prompt(
            agent_id=self.agent_id if is_collab else None,
            agents=self.agents if is_collab else None,
            messaging_enabled=self.messaging_enabled,
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> AgentState:
        system_msg = Message(role="system", content=self._system_prompt)
        return AgentState(
            system_messages=[system_msg],
            messages=list(message_history) if message_history else [],
            metadata={"agent_id": self.agent_id},
        )

    def get_task_message(self, task_description: str) -> str:
        """Render the instance template with task description."""
        is_collab = len(self.agents) > 1
        return _render_instance_template(
            task=task_description,
            agent_id=self.agent_id if is_collab else None,
            agents=self.agents if is_collab else None,
            messaging_enabled=self.messaging_enabled,
        )

    def get_solo_task_message(self, descriptions: Dict[str, str]) -> str:
        """Render a combined task message for solo setting (both features)."""
        combined = []
        for fid in sorted(descriptions.keys(), key=str):
            combined.append(f"## Feature {fid}\n\n{descriptions[fid]}")
        task_text = "\n\n---\n\n".join(combined)
        return self.get_task_message(task_text)

    def build_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        messages = []
        for msg in state.system_messages:
            messages.append(msg.to_dict())
        for msg in state.messages:
            messages.append(msg.to_dict())
        return messages

    def parse_response(
        self, text: str, state: AgentState,
    ) -> Tuple[Message, AgentState]:
        msg = Message(role="assistant", content=text)
        new_state = state.add_message(msg)
        return msg, new_state

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        if message.content and STOP_SIGNAL in message.content:
            return True
        return False

    def set_seed(self, seed: int) -> None:
        pass

    def stop(self, last_message: Optional[Message], state: AgentState) -> None:
        pass
