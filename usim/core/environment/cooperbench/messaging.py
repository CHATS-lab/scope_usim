"""Messaging connector for inter-agent communication via Redis.

Wraps CooperBench's Redis-based messaging pattern. Each agent has an inbox
that other agents push messages to.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessagingConnector:
    """Redis-based mailbox messaging between agents.

    Reuses CooperBench's messaging pattern: each agent has a Redis list
    as inbox, messages are pushed/popped as JSON objects.
    """

    def __init__(
        self,
        agent_id: str,
        agents: List[str],
        url: str = "redis://localhost:6379",
    ):
        """Initialize messaging connector.

        Args:
            agent_id: This agent's unique identifier
            agents: List of all agent IDs in the collaboration
            url: Redis URL. Supports namespacing via #prefix
        """
        import redis

        self.agent_id = agent_id
        self.agents = agents

        # Parse optional namespace prefix from URL (format: url#prefix)
        if "#" in url:
            url, self._prefix = url.split("#", 1)
            self._prefix += ":"
        else:
            self._prefix = ""

        self._client = redis.from_url(url)
        self._inbox_key = f"{self._prefix}{agent_id}:inbox"

        # Clear stale messages from previous runs
        self._client.delete(self._inbox_key)

    def send(self, recipient: str, content: str) -> None:
        """Send a message to another agent's inbox.

        Args:
            recipient: Target agent's ID
            content: Message content
        """
        message = {
            "from": self.agent_id,
            "to": recipient,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self._client.rpush(f"{self._prefix}{recipient}:inbox", json.dumps(message))

    def receive(self) -> List[Dict[str, Any]]:
        """Get all pending messages from inbox (empties the inbox).

        Returns:
            List of message dicts with from, to, content, timestamp
        """
        messages = []
        while True:
            msg = self._client.lpop(self._inbox_key)
            if msg is None:
                break
            messages.append(json.loads(msg))
        return messages

    def peek(self) -> int:
        """Check how many messages are waiting without consuming them.

        Returns:
            Number of pending messages
        """
        return self._client.llen(self._inbox_key)
