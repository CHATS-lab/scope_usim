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

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

Example of a CORRECT response:
<example_response>
I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

[Makes bash tool call with {"command": "ls -la"} as arguments]
</example_response>

<system_information>
%s
</system_information>

## Useful command examples

### Create a new file:

```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```bash
anything
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
    prompt = "You are a helpful assistant that can interact with a computer.\n"
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
        text += "<important>You have LIMITED context. Be efficient — start writing code early.</important>\n\n"
        text += "1. Quickly identify the relevant files (1-2 commands max)\n"
        text += "2. Implement the feature by editing/creating the necessary files\n"
        text += "3. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
        text += "   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>\n"

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
