"""
Milestone 1 agent entrypoint.

The active implementation keeps the local-model + sub-agent architecture, but
milestone-1 scope only:
- coordinator + specialists
- no active custom tools
- no benchmark-specific logic
"""

from .specialists import root_agent

__all__ = ["root_agent"]
