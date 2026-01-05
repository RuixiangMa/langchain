"""Runnable wrapper for prefix-preserved chat history.

This module provides a Runnable wrapper that manages prefix-preserved chat history,
optimizing for KV cache utilization in inference engines that support prefix matching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.utils import ConfigurableFieldSpec

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

from langchain_core.prefix_preserved import PrefixPreservedChatHistory


class RunnableWithPrefixPreservedHistory(RunnableBindingBase):
    """Runnable wrapper that manages prefix-preserved chat history.

    This wrapper ensures that the system prefix remains stable across all requests,
    maximizing KV cache hit rates in inference engines that support prefix matching.

    """

    get_session_history: Callable[..., BaseChatMessageHistory]

    def __init__(
        self,
        runnable: Runnable,
        get_session_history: Callable[..., BaseChatMessageHistory],
        *,
        enable_prefix_preserved: bool = True,
        system_prefix: Sequence[BaseMessage] | None = None,
        get_session_system_prefix: Callable[..., Sequence[BaseMessage]] | None = None,
        input_messages_key: str | None = None,
        history_messages_key: str | None = None,
        history_factory_config: Sequence[ConfigurableFieldSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the RunnableWithPrefixPreservedHistory.

        Args:
            runnable: The runnable to wrap.
            get_session_history: Function that returns a chat message history
                for a given session.
            enable_prefix_preserved: Whether to enable prefix preservation.
            system_prefix: Fixed system messages that always appear first.
            get_session_system_prefix: Optional function that returns session-specific
                system prefix. If provided, takes precedence over system_prefix.
            input_messages_key: Key in input dict that contains the new messages.
            history_messages_key: Key in input dict that should contain the history.
            history_factory_config: Configuration specs for the history factory.
            **kwargs: Additional kwargs to pass to parent class.
        """
        self._enable_prefix_preserved = enable_prefix_preserved
        self._default_system_prefix = list(system_prefix) if system_prefix else []
        self._get_session_system_prefix = get_session_system_prefix

        def _get_prefix_preserved_history(**kwargs: Any) -> PrefixPreservedChatHistory:
            base_history = get_session_history(**kwargs)

            if isinstance(base_history, PrefixPreservedChatHistory):
                return base_history

            session_system_prefix = self._default_system_prefix
            if self._get_session_system_prefix is not None:
                session_system_prefix = list(self._get_session_system_prefix(**kwargs))

            return PrefixPreservedChatHistory(
                enable_prefix_preserved=self._enable_prefix_preserved,
                system_prefix=session_system_prefix,
            )

        super().__init__(
            get_session_history=_get_prefix_preserved_history,
            input_messages_key=input_messages_key,
            history_messages_key=history_messages_key,
            history_factory_config=history_factory_config,
            bound=runnable,
            **kwargs,
        )


    def get_session_prefix_preserved_history(
        self, session_id: str, **kwargs: Any
    ) -> PrefixPreservedChatHistory | None:
        """Get the prefix-preserved history for a given session.

        Args:
            session_id: The session ID to get the history for.
            **kwargs: Additional arguments for the history factory.

        Returns:
            The PrefixPreservedChatHistory for the session, or None if not found.
        """
        try:
            get_session_history = self.get_session_history
            history = get_session_history(session_id=session_id, **kwargs)

            if isinstance(history, PrefixPreservedChatHistory):
                return history
        except (AttributeError, Exception):
            pass
        return None
