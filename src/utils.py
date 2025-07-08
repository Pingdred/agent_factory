"""
Agent Factory - Utility Functions

This module contains utility functions for agent execution, including
tool and form execution helpers.

Functions
---------
execute_tool(cat, tool, input)
    Execute a CatTool with the provided input and return the result.
execute_form(cat, form)
    Execute a CatForm and return the result.
"""

from typing import Dict, Any

from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm
from cat.agents.form_agent import FormAgent
from cat.looking_glass.stray_cat import StrayCat
from cat.log import log

from .messages import LLMAction, AgentOutput


def execute_tool(cat: StrayCat, tool: CatTool, input: Dict[str, Any]) -> LLMAction:
    """
    Execute a CatTool with the provided input and return the result.

    Parameters
    ----------
    cat : StrayCat
        The StrayCat instance providing context for the tool execution.
    tool : CatTool
        The tool to execute.
    input : Dict[str, Any]
        The input parameters to pass to the tool function.

    Returns
    -------
    LLMAction
        An action object containing the tool execution results.
    """
    log.debug(f"Executing tool: {tool.name} with input: {input}")
    tool_output = tool.func(
        **input, cat=cat
    )

    # Ensure the output is a string or None, 
    if (tool_output is not None) and (not isinstance(tool_output, str)):
        tool_output = str(tool_output)

    return LLMAction(
        name=tool.name,
        input=input,
        output=tool_output,
        return_direct=tool.return_direct
    )


def execute_form(cat: StrayCat, form: CatForm) -> LLMAction:
    """
    Execute a CatForm and return the result.

    Parameters
    ----------
    cat : StrayCat
        The StrayCat instance providing context for the form execution.
    form : CatForm
        The form class to instantiate and execute.

    Returns
    -------
    LLMAction
        An action object containing the form execution results.
    """
    form_instance = form(cat)
    cat.working_memory.active_form = form_instance

    form_output: AgentOutput = FormAgent().execute(cat)

    return LLMAction(
        name=form.name,
        input=form_instance._model,
        output=form_output.output,
        return_direct=True,  # Forms always return direct output
    )
