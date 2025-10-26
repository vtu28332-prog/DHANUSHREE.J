
# Task 9 — Robust Chatbot example (works with both modern OpenAI SDK and older openai package)
# Put your API key in the environment variable OPENAI_API_KEY before running:
# export OPENAI_API_KEY="sk-..."

import os
import sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: set the OPENAI_API_KEY environment variable and rerun.", file=sys.stderr)
    sys.exit(1)

# Try to use the modern OpenAI client if available; otherwise fall back to the legacy openai package.
client = None
legacy_openai = None
try:
    # Modern SDK style: `from openai import OpenAI` (openai >= 1.0.0)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    try:
        # Legacy SDK: `import openai` (openai < 1.0.0)
        import openai as legacy_openai
        legacy_openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print("ERROR: Could not import any OpenAI library (modern or legacy).", file=sys.stderr)
        raise e

# Conversation state (keeps history)
messages = [{"role": "system", "content": "You are a helpful assistant."}]

def get_reply_from_modern_client(messages):
    # Uses `client.chat.completions.create(...)` form from modern SDK
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    # Safe extraction of assistant content
    try:
        return resp.choices[0].message["content"]
    except Exception:
        # Some SDKs return slightly different shapes; try a couple fallbacks
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)

def get_reply_from_legacy_openai(messages):
    # Uses legacy openai.ChatCompletion.create(...) interface
    resp = legacy_openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message["content"]

def chat_loop():
    print("Chatbot ready. Type messages. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        # Append user message and call API
        messages.append({"role": "user", "content": user_input})
        try:
            if client is not None:
                assistant_text = get_reply_from_modern_client(messages)
            else:
                assistant_text = get_reply_from_legacy_openai(messages)
        except Exception as e:
            print("API error:", e, file=sys.stderr)
            # Remove last user message so next loop doesn't resend it automatically
            messages.pop()
            continue

        # Print and append assistant reply
        print("Bot:", assistant_text.strip())
        messages.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    chat_loop()
