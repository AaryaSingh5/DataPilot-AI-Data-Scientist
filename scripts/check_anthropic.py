"""
Standalone diagnostic script: verify that the Anthropic SDK works with the
configured API key.  Run from the project root with the venv activated:

    python scripts/check_anthropic.py

Expects ANTHROPIC_API_KEY in the environment (or a .env file loaded by you).
"""

import os
import sys
import traceback


def main() -> None:
    # ── 1. Check for API key ─────────────────────────────────────────────
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Set it with:\n"
            '  $env:ANTHROPIC_API_KEY = "sk-ant-api03-..."\n'
            "or add it to a .env file and load it before running this script."
        )
        sys.exit(1)

    print(f"API key found (starts with {api_key[:12]}...)")

    # ── 2. Import SDK ────────────────────────────────────────────────────
    try:
        import anthropic

        print(f"anthropic SDK version: {anthropic.__version__}")
    except ImportError:
        print("ERROR: anthropic package not installed.  pip install anthropic")
        sys.exit(1)

    # ── 3. Model name ────────────────────────────────────────────────────
    model = os.environ.get("DATAPILOT_MODEL", "claude-sonnet-4-5-20250514")
    print(f"Using model: {model}")

    # ── 4. Call the Messages API ─────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=64,
            temperature=0.0,
            system="You are a helpful assistant. Reply in one sentence.",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        # The response text lives at resp.content[0].text
        text = resp.content[0].text
        print(f"\nSUCCESS — Model responded:\n  {text}")
        print(f"\nUsage: input={resp.usage.input_tokens} output={resp.usage.output_tokens}")
    except Exception as exc:
        print(f"\nFAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
