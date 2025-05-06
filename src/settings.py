from enum import Enum
from typing import List, Tuple, Dict

from pydantic import BaseModel, Field, field_validator

from cat.mad_hatter.decorators import plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.looking_glass.prompts import MAIN_PROMPT_PREFIX
from cat.agents.main_agent import MainAgent as CatMainAgent
from cat.log import log

from .agent.base import BaseAgent

def load_allowed_agents(cat: CheshireCat) -> List[Tuple[BaseAgent, str, str]]:
    agents: List = MadHatter().execute_hook(
        "plugin_factory_allowed_agents",
        [],
        cat=cat
    )
    agents.insert(0,(CatMainAgent, "DEFAULT", "Default agent"))
    return agents   

@plugin
def settings_model() -> BaseModel:
    allowed_agents = load_allowed_agents(CheshireCat())
    enum_agents_dict = {agent[1]: agent[2] for agent in allowed_agents}
    AgentEnum = Enum(
        "Agents", 
        enum_agents_dict,
    )

    class SettingsModel(BaseModel):
        agent: AgentEnum = Field(
            title="Agents",
            description="Select the agent to use.",
            default=AgentEnum.DEFAULT,
        )
        max_tools_call: int = Field(
            title="Max tools call (Only for Native FC)",
            default=3,
            description="Maximum number of tools to call in a single interaction.",
        )

        @field_validator("agent", mode='before')
        @classmethod
        def validate_or_default(cls, v):
            if isinstance(v, str) and v in AgentEnum._value2member_map_:
                return AgentEnum(v)
            
            log.error(f"AGENT FACTORY: Agent `{v}` not found, using default agent")
            return AgentEnum.DEFAULT

        @classmethod
        def get_agents(cls) -> Dict[str, BaseAgent]:
            return {
                agent[1]: agent[0] for agent in allowed_agents
            }
           
    return SettingsModel