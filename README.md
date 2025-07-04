# 🏭 Agent Factory for Cheshire Cat AI

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  [![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://) [![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/Pingdred/agent_factory)


**Agent Factory** is a powerful plugin for Cheshire Cat AI that introduces a flexible factory pattern for creating and managing custom agents. Build specialized agents with native LLM Function Calling support, advanced tool integration, and easy memory access.

## 🎯 Why Agent Factory?

- **🔧 Extensible Architecture**: Create specialized agents for different use cases
- **⚡ Native Function Calling**: Built-in support for LLM Function Calling with multiple parameters
- **🧠 Memory Integration**: Full access to Cheshire Cat's episodic and declarative memory
- **🛠️ Tool Management**: Advanced input handling for complex tools
- **🎛️ Easy Management**: Simple UI-based agent selection and configuration

## ✨ Key Features

### 🔌 Extensible Agent System
Create custom agents by inheriting from `LangchainBaseAgent` - build anything from simple chatbots to complex task-oriented agents.

### 🎯 Native Function Calling
Full support for LLM Function Calling with automatic tool binding. Define tools with any number of parameters:
```python
@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, cat) -> float:
    """Calculate distance between two coordinates.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point 
        lat2: Latitude of second point
        lon2: Longitude of second point
    """
    # Your calculation logic here
    return distance
```

### 🧰 Rich Utility Framework
`LangchainBaseAgent` provides powerful methods to interact with Cheshire Cat's ecosystem:
- **Memory Access**: Query episodic and declarative memory
- **Tool Execution**: Execute any registered tool or form
- **Chain Management**: Advanced LLM interaction patterns
- **Result Persistence**: Automatic chat history management

### 🎛️ Easy Management
- UI-based agent selection from the admin panel
- Hot-swappable agents without restarts
- Default fallback to standard Cheshire Cat behavior

## 🚀 Quick Start

### Installation Options

#### 📦 From Admin Panel (Recommended)
1. Open the Cheshire Cat admin panel
2. Navigate to **Plugins** → **Registry**
3. Search for "Agent Factory"
4. Click **Install** and wait for completion
5. The plugin will be automatically activated

## 🏗️ Building Your First Agent

### Step 1: Create Your Agent Class

Create a new Python file in your plugin (e.g., `agents/echo_agent.py`):

```python
from cat.looking_glass.stray_cat import StrayCat
from cat.plugins.agent_factory import LangchainBaseAgent 
from cat.plugins.agent_factory import AgentOutput
from cat.convo.messages import UserMessage

class EchoAgent(LangchainBaseAgent):
    """A simple agent that echoes user messages with a twist."""
    
    def execute(self, cat: StrayCat) -> AgentOutput:
        # Get the user's message
        user_message: UserMessage = cat.working_memory.user_message_json
        
        # Create a response with some personality
        response = f"🔄 Echo: {user_message.text}\n\nDid you know I'm powered by Agent Factory?"
        
        return AgentOutput(output=response)
```

### Step 2: Register Your Agent

In your plugin's main file or `__init__.py`, register the agent:

```python
from typing import List, Tuple
from cat.mad_hatter.decorators import hook
from cat.plugins.agent_factory import LangchainBaseAgent

# Import your custom agent
from .agents.echo_agent import EchoAgent

@hook
def plugin_factory_allowed_agents(agents: List, cat) -> List[Tuple[LangchainBaseAgent, str, str]]:
    """Register custom agents with the factory."""
    agents.append(
        (EchoAgent, "ECHO_AGENT", "Echo Agent - Simple Message Repeater")
    )
    return agents
```

> **💡 Registration Tuple Format:**
> - `AgentClass`: Your agent implementation
> - `"UNIQUE_ID"`: Unique identifier (uppercase recommended)
> - `"Display Name"`: Human-readable name for the UI

### Step 3: Activate Your Agent

1. Go to **Plugins** → **Agent Factory** in the admin panel
2. Select your agent from the dropdown
3. Save the configuration
4. Start chatting with your new agent!

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Cheshire Cat AI                          │
├─────────────────────────────────────────────────────────────┤
│                   Agent Factory Plugin                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   BaseAgent     │  │ LangchainBase   │  │   Custom    │  │
│  │                 │  │     Agent       │  │   Agents    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Function Calling │ Memory Access │ Tool Integration        │
├─────────────────────────────────────────────────────────────┤
│          Cheshire Cat Core (Memory, Tools, Forms)           │
└─────────────────────────────────────────────────────────────┘
```

## 🧰 Advanced Development Guide

### Core Classes & API Reference

#### AgentOutput
```python
class AgentOutput(BaseModelDict):
    output: str | None = None                   # The agent's response text
    actions: List[LLMAction] = []               # List of actions performed
```

#### LLMAction  
```python
class LLMAction(BaseModel):
    id: str | None = None        # Unique action identifier given by the llm provider
    name: str                    # Tool/action name
    input: Dict                  # Parameters for the action
    output: str | None = None    # Action result output
    return_direct: bool = False  # Whether to return output directly
```

### Available Utility Methods

The `LangchainBaseAgent` base class provides powerful utilities:

#### 🔗 LLM Interaction
```python
def run_chain(self, 
              messages: List,
              prompt_prefix: str = "", 
              tools: List[StructuredTool] = None) -> str:
    """Query the LLM with messages and optional tools."""
    pass
```

#### ⚡ Action Execution
```python
def execute_procedure(cat: StrayCat, procedure: CatTool | CatForm, input: Dict) -> LLMAction:
    """Execute a tool or form procedure with the given input."""
    pass

def save_procedure_result(self, action: LLMAction, cat: StrayCat) -> None:
    """Save procedure results to chat history."""
    pass
```

#### 🧠 Memory Access
```python
def get_recalled_procedures_names(self, cat: StrayCat) -> Set[str]:
    """Get names of recalled tools/forms."""
    pass

def get_recalled_episodic_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
    """Access recalled episodic memory as (text, metadata) tuples."""
    pass

def get_recalled_declarative_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
    """Access recalled declarative memory as (text, metadata) tuples."""
    pass

def get_procedures(self, procedures_name: List[str]) -> Dict[str, CatTool | CatForm]:
    """Get procedures by name from the MadHatter."""
    pass
```

## 🔧 Troubleshooting

### Common Issues

#### Agent Not Appearing in Settings
**Problem**: Your agent doesn't show up in the dropdown menu.

**Solutions**:
- Verify the `plugin_factory_allowed_agents` hook is properly implemented
- Check that your agent class inherits from `LangchainBaseAgent`  
- Ensure there are no import errors in your agent file
- Restart the Cheshire Cat instance

#### Function Calling Not Working
**Problem**: Tools aren't being called by the agent.

**Solutions**:
- Verify tools are properly decorated with `@tool`
- Check tool function signatures include the `cat` parameter
- Ensure tools are available in the current context
- Review LLM model supports function calling

#### Memory Access Issues
**Problem**: Agent can't access recalled memories.

**Solutions**:
- Verify memory is populated (check admin panel)
- Ensure proper method calls (`get_recalled_episodic_memory`, etc.)
- Check memory retrieval settings in Cheshire Cat configuration

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
