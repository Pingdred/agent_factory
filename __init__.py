"""
Agent Factory Plugin

A comprehensive plugin for the Cheshire Cat framework that provides enhanced
agent capabilities with LangChain integration, tool calling, and form handling.

Classes
-------
BaseAgent
    Base agent class extending Cheshire Cat's BaseAgent with enhanced functionality.
LangchainBaseAgent
    LangChain-integrated agent with function calling and advanced prompt management.
AgentOutput
    Data structure representing agent execution results with actions.
CatToolMessage
    Message class for representing tool calls in chat history.
LLMAction
    Data structure representing individual tool or function calls.
"""

from .src.agents import BaseAgent, LangchainBaseAgent
from .src.messages import CatToolMessage, LLMAction, AgentOutput

__all__ = [
    "BaseAgent",
    "LangchainBaseAgent", 
    "AgentOutput",
    "CatToolMessage",
    "LLMAction",
]