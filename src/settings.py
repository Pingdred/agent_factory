"""
Agent Factory - Settings Module

This module provides settings configuration for the Agent Factory plugin,
including agent selection and tool call behavior settings.

Functions
---------
load_allowed_agents()
    Load the list of available agents from hooks and include the default agent.
settings_model()
    Create and return the settings model for the plugin configuration.
"""

from enum import Enum
from typing import List, Tuple, Dict

from pydantic import BaseModel, Field, field_validator

from cat.mad_hatter.decorators import plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.agents.main_agent import MainAgent as CatMainAgent
from cat.log import log

from .agents import BaseAgent

def load_allowed_agents() -> List[Tuple[BaseAgent, str, str]]:
    """
    Load the list of available agents from hooks and include the default agent.

    Returns
    -------
    List[Tuple[BaseAgent, str, str]]
        List of tuples where each tuple contains:
        (Agent class, Agent identifier string, Agent display name)
    """
    agents: List = MadHatter().execute_hook(
        "plugin_factory_allowed_agents",
        [],
        cat=CheshireCat()
    )
    agents.insert(0,(CatMainAgent, "DEFAULT", "Default agent"))
    return agents   

@plugin
def settings_model() -> BaseModel:
    """
    Create and return the settings model for the plugin configuration.

    Returns
    -------
    BaseModel
        A Pydantic BaseModel class configured with agent selection and
        tool call behavior settings.
    """
    allowed_agents = load_allowed_agents()
    enum_agents_dict = {agent[1]: agent[2] for agent in allowed_agents}
    AgentEnum = Enum(
        "Agents", 
        enum_agents_dict,
    )

    class SettingsModel(BaseModel):
        """Pydantic model for Agent Factory plugin settings."""
        agent: AgentEnum = Field(
            title="Agents",
            description="Select the agent to use.",
            default=AgentEnum.DEFAULT,
        )
        stream_tool_calls: bool = Field(
            title="Stream Tool Calls",
            default=True,
        )
        notify_tool_calls: bool = Field(
            title="Notify Tool Calls",
            default=False,
            description="Send notifications when a tool call is made."
        )
        
        @field_validator("agent", mode='before')
        @classmethod
        def validate_or_default(cls, v):
            """Validate agent selection and provide fallback to default."""
            if isinstance(v, str) and v in AgentEnum._value2member_map_:
                return AgentEnum(v)
            
            log.error(f"AGENT FACTORY: Agent `{v}` not found, using default agent")
            return AgentEnum.DEFAULT

        @classmethod
        def get_agents(cls) -> Dict[str, BaseAgent]:
            """Get mapping of agent identifiers to agent classes."""
            return {
                agent[1]: agent[0] for agent in allowed_agents
            }
           
    return SettingsModel