"""Example: Using Prefix-Preserved Chat History for Optimal KV Cache Utilization.

This example demonstrates how to use the prefix-preserved chat history
to maximize KV cache hit rates in inference engines that support
prefix matching.

Key concepts:
- System prefix remains stable across all requests
- New messages are appended without modifying the prefix
- This maximizes KV cache reuse for the prefix portion
- Support for extracting system prefix from existing history
- Prefix consistency validation to detect mismatches
- Session-specific system prefixes for different use cases
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "libs" / "core"))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prefix_preserved import (
    PrefixPreservedInMemoryChatHistory,
    PrefixPreservedChatHistory,
    create_prefix_preserved_history,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.prefix import (
    RunnableWithPrefixPreservedHistory,
)


def example_1_basic_usage() -> None:
    """Example 1: Basic usage of PrefixPreservedChatHistory."""
    print("Example 1: Basic Usage")
    print("=" * 50)

    # Define a fixed system prefix
    system_prefix = [
        SystemMessage(content="You are a helpful AI assistant."),
        SystemMessage(content="Always be concise and accurate."),
        SystemMessage(content="If you don't know, say so."),
    ]

    # Create prefix-preserved history
    history = PrefixPreservedInMemoryChatHistory(
        system_prefix=system_prefix,
    )

    # Simulate a conversation
    print("\n--- Conversation Start ---")

    # First interaction
    history.add_messages([
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4"),
    ])

    print(f"Message count: {history.get_total_message_count()}")
    print(f"History count: {history.get_history_message_count()}")
    print(f"Prefix hash: {history.get_prefix_hash()}")

    # Second interaction - prefix remains stable!
    history.add_messages([
        HumanMessage(content="What about 3+3?"),
        AIMessage(content="6"),
    ])

    print(f"\nMessage count: {history.get_total_message_count()}")
    print(f"History count: {history.get_history_message_count()}")
    print(f"Prefix hash: {history.get_prefix_hash()} (should be same)")

    # Show all messages
    print("\n--- Full Message Sequence ---")
    for i, msg in enumerate(history.messages):
        print(f"{i}: [{msg.type}] {msg.content}")

    print()


def example_2_prefix_hash_tracking() -> None:
    """Example 2: Tracking prefix hash for cache optimization."""
    print("Example 2: Prefix Hash Tracking")
    print("=" * 50)

    system_prefix = [SystemMessage(content="You are helpful.")]

    history = PrefixPreservedInMemoryChatHistory(system_prefix=system_prefix)

    # Get initial hash
    initial_hash = history.get_prefix_hash()
    print(f"Initial prefix hash: {initial_hash}")

    # Add messages - hash should stay the same
    history.add_messages([
        HumanMessage(content="Hello"),
        AIMessage(content="Hi!"),
    ])

    after_add_hash = history.get_prefix_hash()
    print(f"After adding messages: {after_add_hash}")
    print(f"Hash unchanged: {initial_hash == after_add_hash}")

    # Update prefix - hash should change
    history.update_system_prefix([SystemMessage(content="You are very helpful.")])

    after_update_hash = history.get_prefix_hash()
    print(f"After updating prefix: {after_update_hash}")
    print(f"Hash changed: {initial_hash != after_update_hash}")

    print()


def example_3_factory_function() -> None:
    """Example 3: Using the factory function."""
    print("Example 3: Factory Function")
    print("=" * 50)

    history = create_prefix_preserved_history(
        system_prefix=[
            SystemMessage(content="System message 1"),
            SystemMessage(content="System message 2"),
        ],
    )

    print(f"Created history with {len(history.system_prefix)} system messages")

    print()


def example_4_messages_without_prefix() -> None:
    """Example 4: Getting messages without the system prefix."""
    print("Example 4: Messages Without Prefix")
    print("=" * 50)

    system_prefix = [
        SystemMessage(content="System 1"),
        SystemMessage(content="System 2"),
    ]

    history = PrefixPreservedInMemoryChatHistory(system_prefix=system_prefix)

    history.add_messages([
        HumanMessage(content="Q1"),
        AIMessage(content="A1"),
        HumanMessage(content="Q2"),
        AIMessage(content="A2"),
    ])

    # Get all messages
    all_messages = history.messages
    print(f"All messages ({len(all_messages)}):")
    for msg in all_messages:
        print(f"  [{msg.type}] {msg.content}")

    # Get only history (without prefix)
    history_only = history.get_messages_without_prefix()
    print(f"\nHistory only ({len(history_only)}):")
    for msg in history_only:
        print(f"  [{msg.type}] {msg.content}")

    print()


def example_5_integration_with_runnable() -> None:
    """Example 5: Integration with Runnable (conceptual)."""
    print("Example 5: Integration with Runnable")
    print("=" * 50)

    # This is a conceptual example showing how to integrate with Runnable
    # In a real scenario, you would use an actual chat model

    # Define system prefix
    system_prefix = [
        SystemMessage(content="You are a helpful assistant."),
        SystemMessage(content="Always provide clear, concise answers."),
    ]

    # Create history
    history = PrefixPreservedInMemoryChatHistory(
        system_prefix=system_prefix,
    )

    # Simulate adding messages
    history.add_messages([
        HumanMessage(content="What is Python?"),
        AIMessage(content="Python is a programming language."),
    ])

    print("Created prefix-preserved history for integration with Runnable")
    print(f"System prefix: {len(system_prefix)} messages")
    print(f"Current history: {history.get_history_message_count()} messages")
    print(f"Prefix hash: {history.get_prefix_hash()}")

    print("\nNote: In production, you would:")
    print("1. Use RunnableWithPrefixPreservedHistory wrapper")
    print("2. Connect to a real chat model (e.g., ChatOpenAI)")
    print("3. Use persistent storage (e.g., Redis) for history")
    print("4. Track prefix hashes for cache optimization")

    print()


def example_8_prefix_consistency_validation() -> None:
    """Example 8: Prefix consistency."""
    print("Example 8: Prefix Consistency")
    print("=" * 50)

    # Case 1: Create with system prefix
    print("Case 1: Create with system prefix")
    history1 = PrefixPreservedChatHistory(
        system_prefix=[SystemMessage(content="You are a helpful assistant.")],
    )
    print(f"✅ Created with prefix: {history1.system_prefix[0].content}")
    print(f"   Prefix hash: {history1.get_prefix_hash()}")

    # Case 2: Disable prefix preservation
    print("\nCase 2: Disable prefix preservation")
    history2 = PrefixPreservedChatHistory(
        enable_prefix_preserved=False,
    )
    print(f"✅ Prefix preservation disabled")
    print(f"   System prefix: {len(history2.system_prefix)} messages")

    print()


def example_9_session_specific_prefixes() -> None:
    """Example 9: Session-specific system prefixes."""
    print("Example 9: Session-Specific System Prefixes")
    print("=" * 50)

    # Define different system prefixes for different sessions
    session_prefixes = {
        "english": [SystemMessage(content="You are an English tutor.")],
        "math": [SystemMessage(content="You are a math tutor.")],
        "science": [SystemMessage(content="You are a science tutor.")],
    }

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        return InMemoryChatMessageHistory()

    def get_session_system_prefix(session_id: str):
        return session_prefixes.get(session_id, [SystemMessage(content="You are a general tutor.")])

    # Create a chain with session-specific prefixes
    # Note: This is a conceptual example
    print("Creating chain with session-specific system prefixes...")
    print("\nSession-specific prefixes:")
    for session_id, prefix in session_prefixes.items():
        print(f"  {session_id}: {prefix[0].content}")

    print("\nWhen users interact with different sessions:")
    print("  - English session gets English tutor prefix")
    print("  - Math session gets Math tutor prefix")
    print("  - Science session gets Science tutor prefix")
    print("  - Each session maintains its own stable prefix for optimal caching")

    print()


def example_9_multiple_system_messages() -> None:
    """Example 9: Handling multiple system messages."""
    print("Example 9: Multiple System Messages")
    print("=" * 50)

    # Create history with multiple system messages
    history = PrefixPreservedChatHistory(
        system_prefix=[
            SystemMessage(content="System instruction 1"),
            SystemMessage(content="System instruction 2"),
            SystemMessage(content="System instruction 3"),
        ],
    )

    # Add conversation messages
    history.add_messages([
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1"),
        SystemMessage(content="System instruction 4"),  # This should be in conversation
        HumanMessage(content="Question 2"),
        AIMessage(content="Answer 2"),
    ])

    print(f"System prefix (first 3 messages):")
    for msg in history.system_prefix:
        print(f"  [{msg.type}] {msg.content}")

    print(f"\nConversation history (includes later system messages):")
    for msg in history.conversation_history:
        print(f"  [{msg.type}] {msg.content}")

    print("\nNote: Only the system_prefix parameter defines the prefix.")
    print("Later system messages added via add_messages are")
    print("included in the conversation history.")

    print()


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 50)
    print("Prefix-Preserved Chat History Examples")
    print("=" * 50)
    print()

    example_1_basic_usage()
    example_2_prefix_hash_tracking()
    example_3_factory_function()
    example_4_messages_without_prefix()
    example_5_integration_with_runnable()
    example_8_prefix_consistency_validation()
    example_9_session_specific_prefixes()
    example_9_multiple_system_messages()

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
