"""Prefix-preserved chat message history for optimal KV cache utilization.

This module provides a chat message history implementation that maintains
a stable prefix to maximize KV cache hit rates in inference engines that
support prefix cache.

The key features are:
- Fixed system prefix that remains at the start of all message sequences
- Append-only context to preserve prefix matching
- Configurable history management while maintaining prefix stability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage

if TYPE_CHECKING:
    from collections.abc import Sequence


class PrefixPreservedChatHistory(BaseChatMessageHistory):
    """Chat message history with preserved prefix for optimal KV cache utilization.

    This implementation ensures that the system prefix (system messages and
    initial context) remains stable across all requests, maximizing the
    effectiveness of KV cache prefix matching in inference engines.

    Key features:
    - System prefix is fixed and always appears first in the message sequence
    - New messages are appended without modifying the prefix
    - Support for extracting system prefix from existing chat history

    Example:
        ```python
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from langchain_core.prefix_preserved import PrefixPreservedChatHistory

        # Create history with a fixed system prefix
        history = PrefixPreservedChatHistory(
            system_prefix=[
                SystemMessage(content="You are a helpful assistant."),
                SystemMessage(content="Always be concise and accurate."),
            ],
        )

        # Add messages - they will be appended after the system prefix
        history.add_messages([
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="4"),
        ])

        # Get all messages - system prefix first, then conversation history
        messages = history.messages
        # [
        #     SystemMessage(content="You are a helpful assistant."),
        #     SystemMessage(content="Always be concise and accurate."),
        #     HumanMessage(content="What is 2+2?"),
        #     AIMessage(content="4"),
        # ]

        # Add more messages - system prefix remains unchanged
        history.add_messages([
            HumanMessage(content="What about 3+3?"),
            AIMessage(content="6"),
        ])
        ```
    """

    def __init__(
        self,
        enable_prefix_preserved: bool = True,
        system_prefix: Sequence[BaseMessage] | None = None,
    ) -> None:
        """Initialize the prefix-preserved chat history.

        Args:
            enable_prefix_preserved: Whether to enable prefix preservation.
                If False, behaves like standard chat history.
            system_prefix: Fixed system messages that always appear first.
        """
        self.enable_prefix_preserved = enable_prefix_preserved

        if not enable_prefix_preserved:
            self.system_prefix = list(system_prefix) if system_prefix else []
            self.conversation_history: list[BaseMessage] = []
            self._all_messages = list(self.system_prefix)
            self._cached_prefix_hash: str | None = None
            return

        self.system_prefix = list(system_prefix) if system_prefix else []
        self.conversation_history: list[BaseMessage] = []
        self._all_messages: list[BaseMessage] = list(self.system_prefix)
        self._cached_prefix_hash: str | None = None

    @property
    def messages(self) -> list[BaseMessage]:
        """Get all messages with system prefix first, followed by conversation history.

        Returns:
            List of messages with system prefix at the start, followed by
            conversation history in chronological order.
        """
        return self._all_messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the conversation history.

        Messages are appended to the conversation history after the system prefix.
        The system prefix is never modified.

        Args:
            messages: Messages to add to the conversation history.
        """
        self.conversation_history.extend(list(messages))
        self._all_messages.extend(list(messages))

    def clear(self) -> None:
        """Clear the conversation history.

        The system prefix is preserved, only the conversation history is cleared.
        """
        self.conversation_history.clear()
        self._all_messages = list(self.system_prefix)

    def update_system_prefix(self, new_prefix: Sequence[BaseMessage]) -> None:
        """Update the system prefix.

        Warning: This will invalidate any existing KV cache entries that
        depend on the previous prefix. Use with caution.

        Args:
            new_prefix: New system prefix to use.
        """
        self._cached_prefix_hash = None
        self.system_prefix = list(new_prefix)
        self._all_messages = self.system_prefix + self.conversation_history

    def get_prefix_hash(self) -> str:
        """Get a hash of the system prefix for cache key generation.

        This can be used to generate cache keys that are stable as long as
        the system prefix remains unchanged.

        Returns:
            A string hash representing the system prefix.
        """
        if self._cached_prefix_hash is None:
            import hashlib
            prefix_str = "".join(str(msg) for msg in self.system_prefix)
            self._cached_prefix_hash = hashlib.md5(prefix_str.encode()).hexdigest()
        return self._cached_prefix_hash

    def get_messages_without_prefix(self) -> list[BaseMessage]:
        """Get only the conversation history without the system prefix.

        Returns:
            List of conversation messages (excluding system prefix).
        """
        return list(self.conversation_history)

    def get_total_message_count(self) -> int:
        """Get the total number of messages (including system prefix).

        Returns:
            Total message count.
        """
        return len(self.system_prefix) + len(self.conversation_history)

    def get_history_message_count(self) -> int:
        """Get the number of conversation messages (excluding system prefix).

        Returns:
            Number of conversation messages.
        """
        return len(self.conversation_history)


class PrefixPreservedInMemoryChatHistory(PrefixPreservedChatHistory):
    """In-memory implementation of prefix-preserved chat history.

    This is a convenience class that combines PrefixPreservedChatHistory with
    in-memory storage. For production use, consider using a persistent
    implementation backed by Redis, database, or other storage.

    Example:
        ```python
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prefix_preserved import PrefixPreservedInMemoryChatHistory

        history = PrefixPreservedInMemoryChatHistory(
            system_prefix=[SystemMessage(content="You are a helpful assistant.")],
        )

        history.add_messages([HumanMessage(content="Hello!")])
        print(history.messages)
        ```
    """

    pass


def create_prefix_preserved_history(
    enable_prefix_preserved: bool = True,
    system_prefix: Sequence[BaseMessage] | None = None,
) -> PrefixPreservedChatHistory:
    """Factory function to create a prefix-preserved chat history.

    Args:
        enable_prefix_preserved: Whether to enable prefix preservation.
            If False, behaves like standard chat history.
        system_prefix: Fixed system messages that always appear first.

    Returns:
        A new PrefixPreservedChatHistory instance.

    Example:
        ```python
        from langchain_core.messages import SystemMessage
        from langchain_core.prefix_preserved import create_prefix_preserved_history

        history = create_prefix_preserved_history(
            enable_prefix_preserved=True,
            system_prefix=[SystemMessage(content="You are a helpful assistant.")],
        )
        ```
    """
    return PrefixPreservedChatHistory(
        enable_prefix_preserved=enable_prefix_preserved,
        system_prefix=list(system_prefix) if system_prefix else [],
    )
