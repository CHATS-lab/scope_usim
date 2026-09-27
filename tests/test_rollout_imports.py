"""Import checks for the rollout modules and every in-repo ``from usim... import``.

The rollout functions import some helpers lazily inside function bodies (for
example the evaluation path of the co-training rollouts), so a broken name only
fails when that branch runs on a GPU node. The static test below resolves every
``from usim.<module> import <name>`` statement in the package and the entry
scripts, including nested ones, against the target module's source without
importing anything. The dynamic test imports each rollout module when Slime is
installed.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

ROLLOUT_MODULES = [
    "usim.slime.rollout",
    "usim.slime.cotrain_rollout",
    "usim.slime.tau2_cotrain_rollout",
    "usim.p4g.rollout",
    "usim.cooperbench.rollout",
]


def _module_file(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _top_level_names(path: Path) -> Set[str]:
    """Names bound at module level: defs, classes, assignments, imports."""
    tree = ast.parse(path.read_text())
    names: Set[str] = set()
    for node in tree.body:
        nodes = [node]
        # Names bound inside module-level try/if blocks count too.
        if isinstance(node, (ast.Try, ast.If)):
            nodes = list(ast.walk(node))
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    names.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for alias in n.names:
                    names.add((alias.asname or alias.name).split(".")[0])
    # PEP 562 lazy attributes: a module-level __getattr__ serves the names
    # it lists in __all__.
    if "__getattr__" in names:
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                names.update(
                    e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return names


def unresolved_usim_imports(root: Path) -> List[Tuple[str, int, str, str]]:
    """Return (file, line, module, name) for every unresolvable usim import."""
    sources = sorted((root / "usim").rglob("*.py")) + sorted(root.glob("train_*.py"))
    cache: Dict[Path, Set[str]] = {}
    problems = []
    for src in sources:
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                continue
            if node.module != "usim" and not node.module.startswith("usim."):
                continue
            target = _module_file(root, node.module)
            rel = str(src.relative_to(root))
            if target is None:
                problems.append((rel, node.lineno, node.module, "<module>"))
                continue
            names = cache.setdefault(target, _top_level_names(target))
            for alias in node.names:
                if alias.name == "*" or alias.name in names:
                    continue
                if _module_file(root, f"{node.module}.{alias.name}") is not None:
                    continue  # importing a submodule
                problems.append((rel, node.lineno, node.module, alias.name))
    return problems


def test_every_usim_import_resolves():
    assert unresolved_usim_imports(REPO_ROOT) == []


def _launcher_function_paths() -> Set[str]:
    paths: Set[str] = set()
    pattern = re.compile(r"--(?:rollout-function-path|data-source-path)\s+(usim\.[\w.]+)")
    for script in (REPO_ROOT / "cmd").rglob("*.sh"):
        paths.update(pattern.findall(script.read_text()))
    return paths


@pytest.mark.parametrize("dotted", sorted(_launcher_function_paths()))
def test_launcher_function_paths_exist(dotted):
    module, _, attr = dotted.rpartition(".")
    target = _module_file(REPO_ROOT, module)
    assert target is not None, f"{dotted}: module {module} not found"
    assert attr in _top_level_names(target), f"{dotted}: {attr} not defined in {module}"


@pytest.mark.parametrize("module", ROLLOUT_MODULES)
def test_rollout_module_imports(module):
    # A bare "slime" import can resolve to the uninstalled submodule directory
    # (a namespace package), so probe a real Slime module instead.
    pytest.importorskip("slime.rollout.base_types", reason="Slime is not installed")
    pytest.importorskip("tau2.run", reason="tau2-bench is not installed")
    if module != "usim.p4g.rollout":
        # usim.core.environment.tau2 parses tool calls with SGLang.
        pytest.importorskip("sglang.srt.function_call.function_call_parser", reason="SGLang is not installed")
    if module == "usim.cooperbench.rollout":
        pytest.importorskip("cooperbench.eval.sandbox", reason="CooperBench is not installed")
    importlib.import_module(module)
