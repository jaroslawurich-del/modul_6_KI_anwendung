import ollama

# Store the conversation history
messages = []

def chat(message):
    """
    Send a message to the Ollama model while preserving conversation history.
    """
    global messages

    # Add the user's message to the history
    messages.append({
        "role": "user",
        "content": message
    })

    # Send the entire conversation history to the model
    response = ollama.chat(
        model="llama3.2",  # Change to your installed model if needed
        messages=messages
    )

    # Extract the assistant's reply
    assistant_reply = response["message"]["content"]

    # Add the assistant's reply to the history
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    # Print the reply
    print(f"\nAssistant: {assistant_reply}\n")


def main():
    print("=== Ollama Chat ===")
    print("Type 'exit', 'quit', or 'q' to end the chat.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        chat(user_input)


if __name__ == "__main__":
    main()