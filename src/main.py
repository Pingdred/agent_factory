from typing import List, Tuple

from cat.mad_hatter.decorators import hook
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.agents.main_agent import MainAgent as CatMainAgent
from cat.log import log

from .agent.base import BaseAgent
from .agent.main_agent import MainAgent as NativeFcAgent
from .settings import BaseSettings, load_allowed_agents

@hook
def plugin_factory_allowed_agents(agents: List[Tuple[BaseAgent, str, str]], cat) -> list:
    agents.extend([
        (CatMainAgent, "DEFAULT", "Default agent"),
        (NativeFcAgent, "STANDARD_WITH_FC", "Default with native function calling")
    ])
    return agents

def _set_agent() -> None:
    cat = CheshireCat()
    allowed_agents = load_allowed_agents(cat)
    # Key is the agent description (Enum value), value is the agent class
    agents_dict = {agent[2]: agent[0] for agent in allowed_agents}
    agents_names = {agent[2]: agent[1] for agent in allowed_agents}

    settings = BaseSettings(**cat.mad_hatter.get_plugin().load_settings())
    SelectedAgent: BaseAgent = agents_dict.get(settings.agent, CatMainAgent)

    if isinstance(cat.main_agent, SelectedAgent):
        return
    
    log.debug(f"AGENT FACTORY: Setting agent to {agents_names[settings.agent]}")
    cat.main_agent = SelectedAgent()

@hook 
def before_cat_reads_message(_, cat):
    _set_agent()