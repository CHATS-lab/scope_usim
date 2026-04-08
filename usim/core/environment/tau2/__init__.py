"""Tau2-bench environment for USIM.

Uses PEP 562 lazy ``__getattr__`` so importing a submodule (e.g.
``vs_parsing``) does not trigger the heavy ``environment.py`` import chain
(which pulls in sglang + tau2). Backwards-compatible: callers that do
``from usim.core.environment.tau2 import Tau2Environment`` still work.
"""


def __getattr__(name):
    if name == "Tau2Environment":
        from usim.core.environment.tau2.environment import Tau2Environment
        return Tau2Environment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Tau2Environment"]
