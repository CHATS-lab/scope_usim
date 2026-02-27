"""CooperBench agent implementing BaseAgent protocol.

Renders system/instance prompts from CooperBench mini-swe-agent-v2 templates
for coding agents that use OpenAI-style tool calling (bash + optional send_message).
"""

import logging
import platform
from typing import Any, Dict, List, Optional, Tuple

from usim.core.types import AgentState, Message

logger = logging.getLogger(__name__)

STOP_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# System template — v2 style (from CooperBench mini_swe_agent_v2/config/mini.yaml)
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.
{collaboration_block}\
"""

COLLABORATION_INTRO = """\
You are {agent_id} working as a team with: {agents_str}.
You are all working on related features in the same codebase. Each agent has their own workspace.
{messaging_line}"""

COLLABORATION_DETAIL = """\
<collaboration>
Each agent has their own workspace. At the end, all agents' changes will be merged together.
**Important**: Coordinate to avoid merge conflicts - your patches must cleanly combine!
{messaging_section}
Submit when your feature is complete: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
</collaboration>"""

MESSAGING_SECTION = """\
## Messaging
Use the `send_message` tool with `recipient` and `content` arguments to coordinate with teammates.
Messages from teammates appear as: [Message from <agent_name>]: ..."""

# Instance template — v2 style (tool calling, not bash blocks)
INSTANCE_TEMPLATE = """\
Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.
{collaboration_detail}
## Workflow

1. Explore the codebase and find relevant files for your feature
{messaging_step}3. Implement your changes
4. Test your changes work
5. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine the submit command with any other command. \
<important>After this command, you cannot continue working on this task.</important>

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
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or \
write/load environment variables from files
- Submit your changes and finish your work by issuing the following command: \
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
  Do not combine it with any other command. \
<important>After this command, you cannot continue working on this task.</important>

<system_information>
{system} {release} {version} {machine}
</system_information>"""


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
        """Initialize the CooperBench agent.

        Args:
            agent_id: This agent's unique identifier
            agents: List of all agent IDs (for collaboration)
            messaging_enabled: Whether messaging is enabled
            setting: Training setting — "baseline", "solo", or "coop"
        """
        self.agent_id = agent_id
        self.setting = setting
        # For baseline/solo, disable collaboration (single agent)
        if setting in ("baseline", "solo"):
            self.agents = [agent_id]
            self.messaging_enabled = False
        else:
            self.agents = agents or [agent_id]
            self.messaging_enabled = messaging_enabled
        self._system_prompt = self._render_system_prompt()

    def _render_system_prompt(self) -> str:
        """Render the system prompt from templates."""
        is_collab = len(self.agents) > 1

        if is_collab:
            messaging_line = "Use the send_message tool to coordinate with teammates." if self.messaging_enabled else ""
            collaboration_block = COLLABORATION_INTRO.format(
                agent_id=self.agent_id,
                agents_str=", ".join(self.agents),
                messaging_line=messaging_line,
            )
        else:
            collaboration_block = ""

        return SYSTEM_TEMPLATE.format(collaboration_block=collaboration_block)

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> AgentState:
        """Initialize agent state with system prompt."""
        system_msg = Message(role="system", content=self._system_prompt)
        return AgentState(
            system_messages=[system_msg],
            messages=list(message_history) if message_history else [],
            metadata={"agent_id": self.agent_id},
        )

    def get_task_message(self, task_description: str) -> str:
        """Render the instance template with task description.

        Args:
            task_description: Feature description / task text

        Returns:
            Rendered instance message content
        """
        is_collab = len(self.agents) > 1

        if is_collab:
            messaging_section = MESSAGING_SECTION if self.messaging_enabled else ""
            collaboration_detail = COLLABORATION_DETAIL.format(messaging_section=messaging_section)
            messaging_step = (
                "2. Use the send_message tool to tell teammates which files you plan to modify\n"
                if self.messaging_enabled
                else ""
            )
        else:
            collaboration_detail = ""
            messaging_step = ""

        return INSTANCE_TEMPLATE.format(
            task=task_description,
            collaboration_detail=collaboration_detail,
            messaging_step=messaging_step,
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
        )

    def get_solo_task_message(self, descriptions: Dict[str, str]) -> str:
        """Render a combined task message for solo setting (both features).

        Args:
            descriptions: Dict mapping feature_id (str) to description text

        Returns:
            Rendered instance message with both features combined
        """
        combined = []
        for fid in sorted(descriptions.keys(), key=str):
            combined.append(f"## Feature {fid}\n\n{descriptions[fid]}")
        task_text = "\n\n---\n\n".join(combined)
        return self.get_task_message(task_text)

    def build_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        """Build messages for LLM generation."""
        messages = []
        for msg in state.system_messages:
            messages.append(msg.to_dict())
        for msg in state.messages:
            messages.append(msg.to_dict())
        return messages

    def parse_response(
        self, text: str, state: AgentState,
    ) -> Tuple[Message, AgentState]:
        """Parse agent response text into a Message and update state."""
        msg = Message(role="assistant", content=text)
        new_state = state.add_message(msg)
        return msg, new_state

    @classmethod
    def is_stop(cls, message: Message) -> bool:
        """Check if message contains the stop signal."""
        if message.content and STOP_SIGNAL in message.content:
            return True
        return False

    def set_seed(self, seed: int) -> None:
        """Set random seed (no-op for this agent)."""
        pass

    def stop(self, last_message: Optional[Message], state: AgentState) -> None:
        """Called when session ends."""
        pass
