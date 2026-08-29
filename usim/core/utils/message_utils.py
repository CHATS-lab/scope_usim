"""Message manipulation utilities."""

from typing import Any, Dict, List, Literal

from usim.core.types import Message


def flip_message_roles(
    messages: List[Message],
    flip_map: Dict[str, str] = None,
) -> List[Message]:
    """Flip message roles according to a mapping.

    Default mapping flips user/assistant for user simulator perspective.

    Args:
        messages: Messages to flip
        flip_map: Role mapping (default: user<->assistant)

    Returns:
        New list with flipped roles
    """
    if flip_map is None:
        flip_map = {
            "user": "assistant",
            "assistant": "user",
        }

    flipped = []
    for msg in messages:
        new_role = flip_map.get(msg.role, msg.role)
        flipped.append(Message(
            role=new_role,
            content=msg.content,
            tool_calls=msg.tool_calls,
            tool_call_id=msg.tool_call_id,
            timestamp=msg.timestamp,
        ))

    return flipped


def messages_to_dict_list(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert list of Message objects to dict format.

    Args:
        messages: Message objects to convert

    Returns:
        List of message dicts in OpenAI format
    """
    return [msg.to_dict() for msg in messages]


def filter_messages_by_role(
    messages: List[Message],
    roles: List[Literal["user", "assistant", "system", "tool"]],
) -> List[Message]:
    """Filter messages to only include specified roles.

    Args:
        messages: Messages to filter
        roles: Roles to keep

    Returns:
        Filtered message list
    """
    return [msg for msg in messages if msg.role in roles]


def extract_text_content(messages: List[Message]) -> str:
    """Extract all text content from messages.

    Args:
        messages: Messages to extract from

    Returns:
        Concatenated text content
    """
    parts = []
    for msg in messages:
        if msg.content:
            parts.append(f"[{msg.role}]: {msg.content}")
    return "\n".join(parts)


def count_tokens_in_messages(
    messages: List[Dict[str, Any]],
    tokenizer: Any,
) -> int:
    """Count tokens in a message list.

    Args:
        messages: Messages in dict format
        tokenizer: Tokenizer with apply_chat_template

    Returns:
        Total token count
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))
