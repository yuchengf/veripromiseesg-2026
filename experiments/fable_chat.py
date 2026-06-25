"""Run the Fable 5 system prompt on the Opus model — a minimal streaming chat REPL.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 fable_chat.py                 # defaults to claude-opus-4-8
    python3 fable_chat.py claude-fable-5  # or test against the real Fable model

Type 'exit' / Ctrl-D to quit.
"""
import sys
import pathlib
import anthropic

MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-8"
PROMPT_FILE = pathlib.Path(__file__).with_name("fable5_prompt.md")

system_prompt = PROMPT_FILE.read_text(encoding="utf-8")
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# cache_control on the big system block → only billed full price once per session
system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

print(f"[model: {MODEL} | system prompt: {len(system_prompt):,} chars]\n")

messages = []
while True:
    try:
        user = input("you> ").strip()
    except EOFError:
        break
    if user.lower() in {"exit", "quit"}:
        break
    if not user:
        continue

    messages.append({"role": "user", "content": user})
    print("fable> ", end="", flush=True)
    reply = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            reply += text
    print("\n")
    messages.append({"role": "assistant", "content": reply})
