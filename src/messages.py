"""
Agent Factory - Message Classes

This module contains message and data structure classes used by agents
for communication and tool calling functionality.
"""

from typing import Dict, List, Tuple

from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage, ToolCall

from cat.convo.messages import CatMessage
from cat.utils import BaseModelDict


class LLMAction(BaseModel):
    """Represents an action (tool call) requested by the LLM."""
    id: str | None = None
    name: str
    input: Dict
    output: str | None = None
    return_direct: bool = False


class AgentOutput(BaseModelDict):
    output: str | None = None
    actions: List[LLMAction] = []

    @property
    def intermediate_steps(self) -> List[Tuple[Tuple[str, Dict], str]]:
        """Return the list of actions as intermediate steps. Used for compatibility with ChesireCat core."""
        return [((action.name, action.input), action.output) for action in self.actions]


class CatToolMessage(CatMessage):
    """Message representing a tool call and its result in the chat history."""
    action: LLMAction

    def langchainfy(self) -> List[ToolMessage]:
        """Convert to LangChain format for chat history."""
        # Message to represent the tool called
        tool_call = AIMessage(
            content=f"Tool Call: {self.action.name}",
            tool_calls=[
                ToolCall(
                    name=self.action.name,
                    args=self.action.input,
                    id=self.action.id,
                )
            ],
        )

        # Message to represent the result of the tool called
        tool_result = ToolMessage(
            content=self.action.output,
            tool_call_id=self.action.id,
        )

        return [tool_call, tool_result]
