"""Tests for prefix-preserved chat history."""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prefix_preserved import (
    PrefixPreservedChatHistory,
    PrefixPreservedInMemoryChatHistory,
    create_prefix_preserved_history,
)


def test_prefix_preserved_history_basic() -> None:
    """Test basic prefix-preserved history functionality."""
    system_prefix = [
        SystemMessage(content="You are a helpful assistant."),
        SystemMessage(content="Always be concise."),
    ]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    # Add some messages
    history.add_messages([
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there!"),
    ])

    # Check that system prefix is first
    messages = history.messages
    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "You are a helpful assistant."
    assert isinstance(messages[1], SystemMessage)
    assert messages[1].content == "Always be concise."
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "Hello!"
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "Hi there!"


def test_prefix_preserved_history_append_only() -> None:
    """Test that messages are appended without modifying the prefix."""
    system_prefix = [SystemMessage(content="System prompt")]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    # Add first batch
    history.add_messages([
        HumanMessage(content="First question"),
        AIMessage(content="First answer"),
    ])

    # Add second batch
    history.add_messages([
        HumanMessage(content="Second question"),
        AIMessage(content="Second answer"),
    ])

    messages = history.messages
    assert len(messages) == 5
    # System prefix should still be first
    assert messages[0].content == "System prompt"
    # Messages should be in order
    assert messages[1].content == "First question"
    assert messages[2].content == "First answer"
    assert messages[3].content == "Second question"
    assert messages[4].content == "Second answer"


def test_prefix_preserved_history_clear() -> None:
    """Test that clear preserves system prefix."""
    system_prefix = [SystemMessage(content="System prompt")]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    history.add_messages([
        HumanMessage(content="Question"),
        AIMessage(content="Answer"),
    ])

    # Clear history
    history.clear()

    # System prefix should remain
    messages = history.messages
    assert len(messages) == 1
    assert messages[0].content == "System prompt"


def test_prefix_preserved_history_prefix_hash() -> None:
    """Test prefix hash generation."""
    system_prefix = [SystemMessage(content="System prompt")]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    hash1 = history.get_prefix_hash()

    # Add messages - hash should remain the same
    history.add_messages([HumanMessage(content="Question")])

    hash2 = history.get_prefix_hash()

    assert hash1 == hash2

    # Update prefix - hash should change
    history.update_system_prefix([SystemMessage(content="New prompt")])

    hash3 = history.get_prefix_hash()

    assert hash1 != hash3


def test_prefix_preserved_history_get_messages_without_prefix() -> None:
    """Test getting messages without system prefix."""
    system_prefix = [SystemMessage(content="System")]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    history.add_messages([
        HumanMessage(content="Q1"),
        AIMessage(content="A1"),
    ])

    # Get messages without prefix
    history_only = history.get_messages_without_prefix()
    assert len(history_only) == 2
    assert history_only[0].content == "Q1"
    assert history_only[1].content == "A1"


def test_prefix_preserved_history_message_counts() -> None:
    """Test message count methods."""
    system_prefix = [
        SystemMessage(content="S1"),
        SystemMessage(content="S2"),
    ]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    history.add_messages([
        HumanMessage(content="Q1"),
        AIMessage(content="A1"),
    ])

    assert history.get_total_message_count() == 4
    assert history.get_history_message_count() == 2


def test_prefix_preserved_in_memory_history() -> None:
    """Test PrefixPreservedInMemoryChatHistory convenience class."""
    history = PrefixPreservedInMemoryChatHistory(
        system_prefix=[SystemMessage(content="System")],
    )

    history.add_messages([HumanMessage(content="Hello")])

    assert len(history.messages) == 2
    assert history.messages[0].content == "System"


def test_create_prefix_preserved_history_factory() -> None:
    """Test factory function."""
    history = create_prefix_preserved_history(
        system_prefix=[SystemMessage(content="System")],
    )

    assert isinstance(history, PrefixPreservedChatHistory)
    assert len(history.system_prefix) == 1


def test_prefix_preserved_history_validation() -> None:
    """Test validation of system prefix."""
    # Valid prefix
    history = PrefixPreservedChatHistory(
        system_prefix=[SystemMessage(content="System")],
    )
    assert len(history.system_prefix) == 1

    # Invalid prefix (not a list) - should raise error
    with pytest.raises(ValueError, match="system_prefix must be a list"):
        PrefixPreservedChatHistory(system_prefix="not a list")  # type: ignore

    # Invalid prefix (contains non-message) - should raise error
    with pytest.raises(ValueError, match="system_prefix must contain only BaseMessage"):
        PrefixPreservedChatHistory(system_prefix=["not a message"])  # type: ignore


def test_prefix_preserved_history_empty_system_prefix() -> None:
    """Test with empty system prefix."""
    history = PrefixPreservedChatHistory(system_prefix=[])

    history.add_messages([HumanMessage(content="Hello")])

    messages = history.messages
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)


def test_prefix_preserved_history_update_system_prefix() -> None:
    """Test updating system prefix."""
    history = PrefixPreservedChatHistory(
        system_prefix=[SystemMessage(content="Old system")],
    )

    history.add_messages([HumanMessage(content="Question")])

    # Update system prefix
    history.update_system_prefix([SystemMessage(content="New system")])

    messages = history.messages
    assert len(messages) == 2
    assert messages[0].content == "New system"
    assert messages[1].content == "Question"


def test_prefix_preserved_history_multiple_system_messages() -> None:
    """Test with multiple system messages in prefix."""
    system_prefix = [
        SystemMessage(content="System 1"),
        SystemMessage(content="System 2"),
        SystemMessage(content="System 3"),
    ]

    history = PrefixPreservedChatHistory(system_prefix=system_prefix)

    history.add_messages([HumanMessage(content="Question")])

    messages = history.messages
    assert len(messages) == 4
    assert messages[0].content == "System 1"
    assert messages[1].content == "System 2"
    assert messages[2].content == "System 3"
    assert messages[3].content == "Question"
