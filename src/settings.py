from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel, Field

from cat.mad_hatter.decorators import plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.looking_glass.prompts import MAIN_PROMPT_PREFIX
from .agent.base import BaseAgent

BASE_TOOL_PROMPT = "You are a tool agent. You can use the following tools to help the user fulfill their request."

def load_allowed_agents(cat: CheshireCat) -> List[Tuple[BaseAgent, str, str]]:
    return MadHatter().execute_hook(
        "plugin_factory_allowed_agents",
        [],
        cat=cat
    )

class BaseSettings(BaseModel):
    # The field agents is a placeholder for the actual
    # agents Enum field that will be created at runtime
    # in this way the field will be the first in the setings
    agent: str 

    set_system_prompt: bool = Field(
        title="Force system prompt",
        default=False,
        description="Whether to use the following system prompt. This will override the default system prompt and the one from the plugins.",
    )
    system_prompt: str = Field(
        title="System Prompt",
        default=MAIN_PROMPT_PREFIX,
        description="Describe the cheshire cat personality and behavior.",
        extra={"type": "TextArea"},        
    )
    set_tools_prompt: bool = Field(
        title="Force tools prompt",
        default=False,
        description="Whether to use the following tools prompt. This will override the default tools prompt and the one from the plugins.",
    )
    tools_prompt: str = Field(
        title="Tools Prompt",
        default=BASE_TOOL_PROMPT,
        description="Describe the tools available to the cheshire cat.",
        extra={"type": "TextArea"},        
    )
    max_tools_call: int = Field(
        title="Max tools call",
        default=3,
        description="Maximum number of tools to call in a single interaction.",
    )

@plugin
def settings_model() -> BaseModel:
    allowed_agents = load_allowed_agents(CheshireCat())
    agents_dict = {agent[1]: agent[2] for agent in allowed_agents}
    AgentEnum = Enum("Agents", agents_dict)

    class AgentsSettings(BaseSettings):
        """
        Settings for the Cheshire Cat plugin.
        """
        agent: AgentEnum = Field(
            title="Agents",
            description="Select the agent to use.",
            default=AgentEnum.DEFAULT,
        )

    return AgentsSettings