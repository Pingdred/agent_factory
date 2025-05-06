
from typing import Dict, List

from pydantic import BaseModel

from langchain_core.messages import AIMessage, ToolMessage, ToolCall

from cat.agents.base_agent import AgentOutput
from cat.convo.messages import CatMessage

class LLMAction(BaseModel):
    id: str
    name: str
    input: Dict

class CatToolMessage(CatMessage):
    action: LLMAction
    result: AgentOutput

    def langchainfy(self) -> List[ToolMessage]:

        # Message to rapresent the tool called
        tool_call =  AIMessage(
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
            content=self.result.output,
            tool_call_id=self.action.id,
        )

        return [tool_call, tool_result]
