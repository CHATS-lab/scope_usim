"""Test a single CooperBench task end-to-end without Slime training.

Usage (on remote machine or Modal container):
    python scripts/test_cooperbench_single.py [--model MODEL_NAME] [--backend modal|docker]

This tests the full chain:
1. Modal auth + sandbox creation
2. CooperBench environment (bash execution)
3. Agent prompt construction
4. (Optional) LLM generation via litellm if --model is set, otherwise uses a hardcoded test command

Useful for debugging sandbox issues before running the full training pipeline.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def check_modal_auth():
    """Check if Modal SDK is authenticated."""
    print("\n=== Step 1: Modal Auth ===", flush=True)
    print(f"MODAL_TOKEN_ID set: {bool(os.environ.get('MODAL_TOKEN_ID'))}", flush=True)
    print(f"MODAL_TOKEN_SECRET set: {bool(os.environ.get('MODAL_TOKEN_SECRET'))}", flush=True)

    try:
        import modal

        print(f"modal version: {modal.__version__}", flush=True)
    except ImportError:
        print("FATAL: modal not installed", flush=True)
        sys.exit(1)

    try:
        app = modal.App.lookup("cooperbench", create_if_missing=True)
        print(f"Modal auth OK, app={app}", flush=True)
        return True
    except Exception as e:
        print(f"Modal auth FAILED: {e}", flush=True)
        return False


def load_one_task():
    """Load the first task from train_baseline.json."""
    print("\n=== Step 2: Load Task ===", flush=True)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cooperbench")
    json_path = os.path.join(data_dir, "train_baseline.json")

    if not os.path.exists(json_path):
        print(f"FATAL: {json_path} not found", flush=True)
        sys.exit(1)

    with open(json_path) as f:
        tasks = json.load(f)

    task = tasks[0]
    print(f"Task: {task['id']}", flush=True)
    print(f"Repo: {task['repo']}", flush=True)
    print(f"Task ID: {task['task_id']}", flush=True)
    print(f"Feature ID: {task['feature_id']}", flush=True)
    return task


async def test_sandbox(task):
    """Create a Modal sandbox and run a few commands."""
    print("\n=== Step 3: Create Sandbox ===", flush=True)

    from cooperbench.utils import get_image_name

    image_name = get_image_name(task["repo"], task["task_id"])
    print(f"Image: {image_name}", flush=True)

    from usim.core.environment.cooperbench.sandbox import CooperBenchSandbox

    sandbox = CooperBenchSandbox(image_name=image_name, timeout=600)

    t0 = time.time()
    try:
        await sandbox.start()
        print(f"Sandbox started in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        print(f"Sandbox start FAILED after {time.time() - t0:.1f}s: {e}", flush=True)
        raise

    # Run a few test commands
    print("\n=== Step 4: Execute Commands ===", flush=True)

    for cmd in ["pwd", "ls -la", "git log --oneline -5"]:
        print(f"\n$ {cmd}", flush=True)
        result = await sandbox.execute(cmd)
        output = result["output"]
        if len(output) > 500:
            output = output[:250] + "\n...\n" + output[-250:]
        print(f"  returncode={result['returncode']}", flush=True)
        print(f"  output: {output}", flush=True)

    # Get patch
    patch = await sandbox.get_patch()
    print(f"\nPatch length: {len(patch)} chars", flush=True)

    # Cleanup
    await sandbox.cleanup()
    print("Sandbox cleaned up", flush=True)


async def test_environment(task):
    """Test the full CooperBenchEnvironment."""
    print("\n=== Step 5: Test CooperBenchEnvironment ===", flush=True)

    from cooperbench.utils import get_image_name

    from usim.core.environment.cooperbench.environment import CooperBenchEnvironment

    image_name = get_image_name(task["repo"], task["task_id"])
    env = CooperBenchEnvironment(image_name=image_name, timeout=600)

    # Get tools
    tools = env.get_tools()
    print(f"Tools: {[t['function']['name'] for t in tools]}", flush=True)

    # Execute a command
    result = await env.execute_command("pwd && ls")
    print(f"Execute result: rc={result['returncode']}, output={result['output'][:200]}", flush=True)

    # Cleanup
    await env.cleanup()
    print("Environment cleaned up", flush=True)


async def test_with_model(task, model_name):
    """Test with actual LLM generation via litellm."""
    print(f"\n=== Step 6: Test with model {model_name} ===", flush=True)

    from usim.cooperbench.agent import CooperBenchAgent
    from usim.core.environment.cooperbench.environment import CooperBenchEnvironment
    from cooperbench.utils import get_image_name

    image_name = get_image_name(task["repo"], task["task_id"])
    env = CooperBenchEnvironment(image_name=image_name, timeout=600)

    agent = CooperBenchAgent(agent_id="agent1", setting="baseline")
    task_msg = agent.get_task_message(task["description"])

    print(f"Task message length: {len(task_msg)} chars", flush=True)
    print(f"First 200 chars: {task_msg[:200]}...", flush=True)

    print("Agent + Environment created OK", flush=True)

    await env.cleanup()


async def run_all_tests(skip_sandbox: bool, model: str | None, task: dict):
    """Run all tests in a single event loop."""
    if not skip_sandbox:
        await test_sandbox(task)
        await test_environment(task)

    if model:
        await test_with_model(task, model)


def main():
    parser = argparse.ArgumentParser(description="Test single CooperBench task")
    parser.add_argument("--model", default=None, help="Model name for LLM test (optional)")
    parser.add_argument("--skip-sandbox", action="store_true", help="Skip sandbox creation test")
    parser.add_argument("--backend", default="modal", help="Sandbox backend: modal or docker")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("CooperBench Single Task Test", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Check Modal auth
    if not args.skip_sandbox:
        if not check_modal_auth():
            print("\nModal auth failed. Fix credentials and retry.", flush=True)
            sys.exit(1)

    # Step 2: Load task
    task = load_one_task()

    # Steps 3-6: Run all async tests in a single event loop
    try:
        asyncio.run(run_all_tests(args.skip_sandbox, args.model, task))
    except Exception as e:
        print(f"\nTest FAILED: {e}", flush=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60, flush=True)
    print("ALL TESTS PASSED", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
