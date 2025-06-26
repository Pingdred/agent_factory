from typing import List, Tuple

from cat.mad_hatter.decorators import hook
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.log import log

from .agent.base import BaseAgent
from .agent.main_agent import MainAgent as NativeFcAgent

@hook
def plugin_factory_allowed_agents(agents: List[Tuple[BaseAgent, str, str]], cat) -> list:
    agents.append(
        (NativeFcAgent, "STANDARD_WITH_FC", "Native Function Calling")
    )
    return agents

def _set_agent() -> None:
    cat = CheshireCat()

    # Load plugin settings
    this_plugin = cat.mad_hatter.get_plugin()
    SettingsModel: type = this_plugin.settings_model()
    settings = SettingsModel(**this_plugin.load_settings())

    # Get the selected agent class from the settings
    allowed_agents = SettingsModel.get_agents()
    SelectedAgent: BaseAgent = allowed_agents.get(settings.agent.name, None)

    # Fallback to default agent if not found
    if SelectedAgent is None:
        log.error(f"AGENT FACTORY: Agent {settings.agent.name} not found, using default agent")
        SelectedAgent = allowed_agents["DEFAULT"]

    # Avoid re-setting the agent if it's already set
    if isinstance(cat.main_agent, SelectedAgent):
        log.debug(f"AGENT FACTORY: Agent already set to {settings.agent.name}")
        return
    
    # Set the selected agent
    log.debug(f"AGENT FACTORY: Setting agent to {settings.agent.name}")
    cat.main_agent = SelectedAgent()

@hook 
def before_cat_reads_message(_, cat):
    _set_agent()