import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-pull local Ollama models.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Ollama model tag to pull. Can be provided more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Models are served by Ollama, so "downloading" a model means `ollama pull`.
    seen: set[str] = set()
    for model in args.model:
        model = model.strip()
        if not model or model in seen:
            continue
        seen.add(model)
        print(f"Pulling Ollama model: {model}")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            print(f"ollama pull failed for {model} (exit {result.returncode}).")
            sys.exit(result.returncode)
        print(f"Ready: {model}")


if __name__ == "__main__":
    main()
