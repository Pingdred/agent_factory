"""
Agent Factory Plugin Main Module

This module handles the registration of agents and manages the agent selection process.
It provides hooks for other plugins to register their custom agents and automatically
sets the selected agent based on plugin settings.
"""

from cat.mad_hatter.decorators import hook
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.log import log

from .agents import BaseAgent


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
