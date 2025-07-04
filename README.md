# Agent Factory for Cheshire Cat AI

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  [![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)


**Agent Factory is a plugin for Cheshire Cat AI that introduces a flexible system for creating and using custom agents.**

This plugin extends Cheshire Cat AI by allowing the development of agents with specific logic, native support for LLM Function Calling, and advanced input management for tools.

## Main Features

  * **Extensible Agents**: Create agents by inheriting from `LangchainBaseAgent`.
  * **Function Calling**: Support for LLM Function Calling, enabling agents to call functions natively. This allows to define tools with any number of parameters. Example:
    ```python
    @tool
    def add_numbers(a: int, b: int, cat) -> int:
        """Adds two numbers.`a` and `b` are the two integers to add"""
        return a + b
    ``` 
  * **Utility Helper**: `LangchainBaseAgent` offers methods to interact with Cheshire Cat AI's memory, tools, etc.

## Installation

### From Cheshire Cat Admin Panel (Recommended)
1. Go to the Cheshire Cat admin panel
2. Navigate to the "Plugins" section  
3. Search for "Agent Factory"
4. Click "Install" to add the plugin to your Cheshire Cat instance

### Manual Installation
1. Clone the repository in the `plugins` directory of your Cheshire Cat instance:
   ```bash
   cd /path/to/cheshire-cat/plugins
   git clone https://github.com/Pingdred/agent_factory.git
   ```
2. Restart Cheshire Cat


## Creating and Registering a New Agent

### Create the Agent
Create a Python class that inherits from `LangchainBaseAgent`:

```python
from cat.looking_glass.stray_cat import StrayCat
from cat.plugins.agent_factory import LangchainBaseAgent 
from cat.plugins.agent_factory import AgentOutput

class EchoAgent(LangchainBaseAgent):
    def execute(self, cat: StrayCat) -> AgentOutput:
        user_message: UserMessage = cat.working_memory.user_message_json
        
        return AgentOutput(
            output=user_message.text,
        )
```

### Register the Agent
Use the `plugin_factory_allowed_agents` hook:

```python
from typing import List, Tuple
from cat.mad_hatter.decorators import hook
from cat.plugins.agent_factory import LangchainBaseAgent
from .my_custom_agent_file import MyCustomAgent # Import your agent

@hook
def plugin_factory_allowed_agents(agents: List, cat) -> List[Tuple[LangchainBaseAgent, str, str]]:
    agents.append(
        (MyCustomAgent, "ECHO_AGENT", "Echo Agent")
    )
    return agents
```

Each tuple contains: `(AgentClass, "UNIQUE_NAME", "UI Name")`.

### Set the Agent in the UI

In the **Agent Factory** plugin settings you can select the agent to use, from the list of available agents. The default agent is the standard Cheshire Cat agent.


## Key Utilities in `LangchainBaseAgent`

  * **`run_chain(...)`**: Queries the LLM for responses or actions.
  * **`execute_action(...)`**: Executes a tool or form.
  * **`save_action_result(...)`**: Saves action results to chat history.
  * **`get_recalled_procedures_names(...)`**: Gets names of recalled tools/forms.
  * **`get_recalled_episodic_memory(...)` / `get_recalled_declarative_memory(...)`**: Accesses recalled memory.

