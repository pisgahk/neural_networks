"""
LangChain + Claude Chatbot with Conversation Memory
----------------------------------------------------
A simple CLI chatbot that remembers what you said earlier in the conversation.

Setup:
    pip install langchain langchain-anthropic

Run:
    ANTHROPIC_API_KEY=your_key python chatbot.py
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


def build_chain():
    """Build a conversational chain with a system prompt and message history."""

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful, concise assistant. "
            "You remember the full conversation and can refer back to earlier messages."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # LangChain Expression Language (LCEL): pipe prompt → llm
    chain = prompt | llm
    return chain


def chat():
    chain = build_chain()
    history: list = []

    print("Claude Chatbot (type 'quit' to exit, 'reset' to clear memory)\n")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history.clear()
            print("[Memory cleared]")
            continue

        # Invoke the chain with current input + full history
        response = chain.invoke({
            "input": user_input,
            "history": history,
        })

        assistant_reply = response.content
        print(f"\nClaude: {assistant_reply}")

        # Append to history for next turn
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=assistant_reply))


if __name__ == "__main__":
    chat()
