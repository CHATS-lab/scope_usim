"""Prompt templates for user simulator and agent."""

from typing import Dict, Optional


# User simulator system prompt template
USER_SIM_SYSTEM_PROMPT = """You are simulating a user interacting with a customer service agent.

<simulation_guidelines>
{simulation_guidelines}
</simulation_guidelines>

<scenario>
{instructions}
</scenario>

<communication_rules>
- Stay in character as the user throughout the conversation
- Respond naturally based on your scenario
- Do NOT reveal that you are an AI or simulation
- If your issue is resolved satisfactorily, respond with ###STOP###
- If you need to be transferred to another department, respond with ###TRANSFER###
- If the agent asks about something completely outside your scenario, respond with ###OUT_OF_SCOPE###
</communication_rules>
"""

# Default simulation guidelines
DEFAULT_SIMULATION_GUIDELINES = """
- Act as a realistic customer with the given background and issue
- Express emotions appropriate to the situation (frustration, satisfaction, confusion)
- Ask clarifying questions when needed
- Don't be overly cooperative - push back reasonably if the solution doesn't meet your needs
- Make decisions based on your persona's priorities and constraints
"""

# Agent system prompt template
AGENT_SYSTEM_PROMPT = """You are a helpful customer service agent.

<agent_guidelines>
{agent_guidelines}
</agent_guidelines>

<domain_policy>
{domain_policy}
</domain_policy>

<communication_rules>
- Help the user with their request professionally and efficiently
- Use available tools to look up information or perform actions
- Never make up information - use tools to verify facts
- If you cannot help or the conversation is complete, respond with ###STOP###
- Be concise but thorough in your responses
</communication_rules>
"""

# Default agent guidelines
DEFAULT_AGENT_GUIDELINES = """
- Verify the user's identity before making account changes
- Follow company policies for refunds, exchanges, and modifications
- Escalate complex issues to appropriate departments
- Document all actions taken in the conversation
"""


def get_user_sim_prompt(
    instructions: str,
    simulation_guidelines: Optional[str] = None,
) -> str:
    """Generate a user simulator system prompt.

    Args:
        instructions: User scenario/instructions
        simulation_guidelines: Optional custom guidelines

    Returns:
        Formatted system prompt
    """
    guidelines = simulation_guidelines or DEFAULT_SIMULATION_GUIDELINES
    return USER_SIM_SYSTEM_PROMPT.format(
        simulation_guidelines=guidelines,
        instructions=instructions,
    )


def get_agent_prompt(
    domain_policy: str,
    agent_guidelines: Optional[str] = None,
) -> str:
    """Generate an agent system prompt.

    Args:
        domain_policy: Domain-specific policy
        agent_guidelines: Optional custom guidelines

    Returns:
        Formatted system prompt
    """
    guidelines = agent_guidelines or DEFAULT_AGENT_GUIDELINES
    return AGENT_SYSTEM_PROMPT.format(
        agent_guidelines=guidelines,
        domain_policy=domain_policy,
    )


# Template factory for named templates
TEMPLATE_FACTORY: Dict[str, str] = {
    "user_sim_default": USER_SIM_SYSTEM_PROMPT,
    "agent_default": AGENT_SYSTEM_PROMPT,
}


def register_template(name: str, template: str) -> None:
    """Register a custom template.

    Args:
        name: Template name
        template: Template string
    """
    TEMPLATE_FACTORY[name] = template


def get_template(name: str) -> str:
    """Get a template by name.

    Args:
        name: Template name

    Returns:
        Template string

    Raises:
        KeyError: If template not found
    """
    if name not in TEMPLATE_FACTORY:
        raise KeyError(f"Template '{name}' not found. Available: {list(TEMPLATE_FACTORY.keys())}")
    return TEMPLATE_FACTORY[name]
