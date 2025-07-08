"""
Agent Factory Plugin - Main Module

This module implements the core functionality of the Agent Factory plugin,
managing agent lifecycle and ensuring the correct agent is active for each
conversation. It provides automatic agent switching based on configuration
and includes fallback mechanisms for robustness.

The module uses Cheshire Cat's hook system to integrate seamlessly with
the framework's message processing pipeline.
"""

from cat.mad_hatter.decorators import hook
from cat.looking_glass.cheshire_cat import CheshireCat
from cat.log import log

from .agents import BaseAgent


def _set_agent() -> None:
    """
    Set the active agent based on plugin settings.

    This internal function loads the plugin settings, determines the selected
    agent, and sets it as the main agent for the Cheshire Cat instance.
    It includes fallback logic to use the default agent if the selected
    agent is not found or available.
    """
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
    """
    Hook function called before the cat reads each message.

    This hook ensures that the correct agent is set before processing
    each message, allowing for dynamic agent switching based on
    current plugin settings.
    """
    _set_agent()
