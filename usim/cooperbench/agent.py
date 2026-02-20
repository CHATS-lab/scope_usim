"""CooperBench agent implementing BaseAgent protocol.

Renders system/instance prompts from CooperBench's mini.yaml templates
for coding agents that interact with a bash sandbox.
"""

import logging
import platform
from typing import Any, Dict, List, Optional, Tuple

from usim.core.types import AgentState, Message

logger = logging.getLogger(__name__)

STOP_SIGNAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# System template ported from CooperBench/src/cooperbench/agents/mini_swe_agent/config/mini.yaml
SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.
{collaboration_block}
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>
{collaboration_detail_block}
Failure to follow these rules will cause your response to be rejected."""

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
## Messaging (coordinate with teammates)
Send messages to teammates. Messages appear in their next turn.
```bash
send_message <agent_name> "your message"
```
Messages from teammates appear as: [Message from <agent_name>]: ..."""

# Instance template ported from mini.yaml
INSTANCE_TEMPLATE = """\
Please solve this issue: {task}

You can execute bash commands and edit files to implement the necessary changes.

## Workflow

1. Explore the codebase and find relevant files for your feature
{messaging_workflow_step}3. Implement your changes
4. Test your changes work
5. Submit: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   Do not combine the submit command with any other command. \
<important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or \
write/load environment variables from files

<system_information>
{system} {release} {version} {machine}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. \
Let me check what files are in the current directory to get a better understanding of the codebase.

```bash
ls -la
```
</example_response>"""


class CooperBenchAgent:
    """Agent for CooperBench coding tasks.

    Renders system and instance prompts from CooperBench's templates,
    with support for multi-agent collaboration via messaging.
    """

    def __init__(
        self,
        agent_id: str = "agent1",
        agents: Optional[List[str]] = None,
        messaging_enabled: bool = True,
    ):
        """Initialize the CooperBench agent.

        Args:
            agent_id: This agent's unique identifier
            agents: List of all agent IDs (for collaboration)
            messaging_enabled: Whether messaging is enabled
        """
        self.agent_id = agent_id
        self.agents = agents or [agent_id]
        self.messaging_enabled = messaging_enabled
        self._system_prompt = self._render_system_prompt()

    def _render_system_prompt(self) -> str:
        """Render the system prompt from templates."""
        is_collab = len(self.agents) > 1

        if is_collab:
            messaging_line = "Use send_message to coordinate." if self.messaging_enabled else ""
            collaboration_block = COLLABORATION_INTRO.format(
                agent_id=self.agent_id,
                agents_str=", ".join(self.agents),
                messaging_line=messaging_line,
            )
            messaging_section = MESSAGING_SECTION if self.messaging_enabled else ""
            collaboration_detail_block = COLLABORATION_DETAIL.format(
                messaging_section=messaging_section,
            )
        else:
            collaboration_block = ""
            collaboration_detail_block = ""

        return SYSTEM_TEMPLATE.format(
            collaboration_block=collaboration_block,
            collaboration_detail_block=collaboration_detail_block,
        )

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
        messaging_step = (
            "2. Message teammates about which files you plan to modify\n"
            if self.messaging_enabled and len(self.agents) > 1
            else ""
        )

        return INSTANCE_TEMPLATE.format(
            task=task_description,
            messaging_workflow_step=messaging_step,
            system=platform.system(),
            release=platform.release(),
            version=platform.version(),
            machine=platform.machine(),
        )

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
