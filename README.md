# Agent Factory for Cheshire Cat AI

[](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3Dhttps://%5D\(https://www.google.com/search%3Fq%3Dhttps://\)) [](https://www.google.com/search?q=%5Bhttps://www.google.com/search%3Fq%3Dhttps://%5D\(https://www.google.com/search%3Fq%3Dhttps://\))

**Agent Factory is a plugin for Cheshire Cat AI that introduces a flexible system for creating and using custom agents.**

This plugin extends Cheshire Cat AI by allowing the development of agents with specific logic, native support for LLM Function Calling, and advanced input management for tools.

## Main Features

  * **Extensible Agents**: Create agents by inheriting from `LangchainBaseAgent`.
  * **Native Function Calling Agent**: Includes a re-implementation of the Cheshire Cat AI agent with native function calling support.
  * **Utility Helper**: `LangchainBaseAgent` offers methods to interact with Cheshire Cat AI's memory, tools, etc.

## Installation

Install the plugin from the Cheshire Cat admin panel:
1.  Go to the Cheshire Cat admin panel.
2.  Navigate to the "Plugins" section.
3.  Search for "Agent Factory".
4.  Click "Install" to add the plugin to your Cheshire Cat instance.

Alternatively, you can install it manually:
1.  Clone the repository in the `plugins` directory of your Cheshire Cat instance:
    ```bash
    git clone https://github.com/Pingdred/agent_factory.git
    ```
2. Restart Cheshire Cat.

## Configuration

In the **Agent Factory** plugin settings you can select the agent to use, from the list of available agents. The default agent is the standard Cheshire Cat agent, select the "Native Function Calling" agent to use the new agent with native function calling support.

Additionally, you can set the maximum number of tool calls the "Native Function Calling" agent can make in a single step.

## Creating and Registering a New Agent

**1. Create the Agent:**
Create a Python class that inherits from `LangchainBaseAgent`:

```python
from cat.looking_glass.stray_cat import StrayCat
from cat.plugins.agent_factory import LangchainBaseAgent 
from cat.plugins.agent_factory import AgentOutput

class EchoAgent(LangchainBaseAgent):
    def execute(self, cat: StrayCat) -> AgentOutput:
        user_mesage: UserMessage = cat.working_memory.user_message_json
        
        return AgentOutput(
            output=user_mesage.text,
        )
```

**2. Register the Agent:**
Use the `plugin_factory_allowed_agents` hook:

```python
from typing import List, Tuple
from cat.mad_hatter.decorators import hook
from cat.plugins.agent_factory import LangchainBaseAgent
from .my_custom_agent_file import MyCustomAgent # Import your agent

@hook
def plugin_factory_allowed_agents(agents: List, cat) -> List[Tuple[LangchainBaseAgent, str, str]]:
    agents.append(
        (MyCustomAgent, "MY_CUSTOM_AGENT", "My Custom Agent")
    )
    return agents
```

Each tuple contains: `(AgentClass, "UNIQUE_NAME", "UI Name")`.

### Key Utilities in `LangchainBaseAgent`

  * **`run_chain(...)`**: Queries the LLM for responses or actions.
  * **`execute_action(...)`**: Executes a tool or form.
  * **`save_action_result(...)`**: Saves action results to chat history.
  * **`get_recalled_procedures_names(...)`**: Gets names of recalled tools/forms.
  * **`get_recalled_episodic_memory(...)` / `get_recalled_declarative_memory(...)`**: Accesses recalled memory.

## Included Agents

  * **Default agent (`CatMainAgent`)**: Standard Cheshire Cat agent.
  * **Native Function Calling (`NativeFcAgent`)**: A re-implementation of the Cheshire Cat AI agent with native function calling support.