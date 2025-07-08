"""
Agent Factory Plugin - Agent module

This module provides the base agent classes for the Agent Factory plugin,
including the BaseAgent and LangchainBaseAgent. These classes extend the
Cheshire Cat's BaseAgent with enhanced functionality for tool execution,
form handling, and LangChain integration.
"""

from .base_agent import BaseAgent
from .langchain_agent import LangchainBaseAgent

__all__ = [
    "BaseAgent",
    "LangchainBaseAgent",
]
