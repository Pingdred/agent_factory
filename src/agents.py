"""
Agent Factory - Agent Classes

This module contains all the agent classes for the Agent Factory plugin.
It includes the base agent functionality and the LangChain-based agent implementation.
"""

import inspect
import functools
from typing import Dict, List, Set, Tuple, Callable, Any

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import StructuredTool, Tool
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.mad_hatter.mad_hatter import MadHatter
from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm
from cat.agents.base_agent import BaseAgent as CatBaseAgent
from cat.agents.form_agent import FormAgent
from cat.looking_glass.stray_cat import StrayCat
from cat.looking_glass.callbacks import ModelInteractionHandler, NewTokenHandler
from cat.memory.working_memory import MAX_WORKING_HISTORY_LENGTH
from cat.utils import get_caller_info, langchain_log_output, langchain_log_prompt
from cat.convo.messages import CatMessage, HumanMessage
from cat.log import log

from .messages import LLMAction, CatToolMessage, AgentOutput


# Agent execution utilities
def _execute_tool(cat: StrayCat, tool: CatTool, input: Dict[str, Any]) -> LLMAction:
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


def _execute_form(cat: StrayCat, form: CatForm) -> LLMAction:
    form_instance = form(cat)
    cat.working_memory.active_form = form_instance

    form_output: AgentOutput = FormAgent().execute(cat)

    return LLMAction(
        name=form.name,
        input=form_instance._model,
        output=form_output.output,
        return_direct=True,  # Forms typically return direct output
    )

class BaseAgent(CatBaseAgent):
    """
    Base agent class that extends Cheshire Cat's BaseAgent.
    
    This class provides common functionality for all custom agents including
    procedure execution, memory access, and result saving.
    """

    def __init__(self):
        super().__init__()

    def execute(self, cat: StrayCat) -> AgentOutput:
        """Abstract method to be implemented by the agent."""
        pass

    @staticmethod
    def execute_procedure(procedure: CatTool | CatForm, input: str | Dict, cat: StrayCat, call_id: str| None = None) -> LLMAction:
        """Execute a procedure (tool or form) with the given input."""

        settings = cat.mad_hatter.get_plugin().load_settings()
        if settings.get("stream_tool_calls", True):
            # Stream tool calls to the chat
            cat.send_ws_message(
                f"Executing: `{procedure.name}`\n",
                msg_type="chat_token"
            )

        try:
            if Plugin._is_cat_tool(procedure):
                res = _execute_tool(cat, procedure, input)
                return LLMAction(call_id=call_id, **res.__dict__)
            
            if Plugin._is_cat_form(procedure):
                return _execute_form(cat, procedure)
            
        except Exception as e:
            log.error(f"Error executing {procedure.procedure_type} `{procedure.name}`: {str(e)}")

        log.error(f"Unknown procedure type: {type(procedure)}")
        raise ValueError(
            f"Unknown procedure type: {type(procedure)}. "
            "Expected CatTool or CatForm."
        )

    def save_procedure_result(self, action: LLMAction, cat) -> None:
        """Save the action result in the chat history."""
        action_call = CatToolMessage(
            user_id=cat.user_id,
            action=action,
        )
        cat.working_memory.update_history(action_call)

    def get_recalled_procedures_names(self, cat: StrayCat) -> Set[str]:
        """Get the names of the recalled procedures from the working memory."""
        recalled_procedures_names = set()
        
        for memory_group in cat.working_memory.procedural_memories:
            memory = memory_group[0]
            metadata = memory.metadata
            
            if (metadata["type"] in ["tool", "form"] and 
                metadata["trigger_type"] in ["description", "start_example"]):
                recalled_procedures_names.add(metadata["source"])
                
        return recalled_procedures_names
    
    def get_recalled_episodic_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """Get the recalled episodic memory from the working memory."""
        return self._memory_points_to_tuple(cat.working_memory.episodic_memories)
    
    def get_recalled_declarative_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """Get the recalled declarative memory from the working memory."""
        return self._memory_points_to_tuple(cat.working_memory.declarative_memories)
    
    def _memory_points_to_tuple(self, memory_points: List[Tuple[Document, ]]) -> List[Tuple[str, Dict]]:
        """Convert memory points to tuples of (text, metadata)."""
        return [
            (m[0].page_content.strip(), m[0].metadata)
            for m in memory_points
        ]

    def get_procedures(self, procedures_name: List[str]) -> Dict[str, CatTool | CatForm]:
        """Get procedures by name from the MadHatter."""
        allowed_procedures: Dict[str, CatTool | CatForm] = {}
        for procedure in MadHatter().procedures:
            if procedure.name in procedures_name:
                allowed_procedures[procedure.name] = procedure
        return allowed_procedures


class LangchainBaseAgent(BaseAgent):
    """
    LangChain-based agent with function calling capabilities.
    
    This agent extends BaseAgent with LangChain integration, providing
    native function calling support and advanced prompt management.
    """

    def run_chain(
        self,
        cat: StrayCat,
        system_prompt: str,
        procedures: Dict[str, CatTool | CatForm] = {},
        chat_history: List[CatMessage | HumanMessage] | None = None,
        max_procedures_calls: int = 1,
        chain_name: str = "Langchain Chain",
    ) -> AgentOutput:
        """
        Run a LangChain chain with function calling capabilities.
        
        Parameters
        ----------
            cat: StrayCat
                The StrayCat instance
            system_prompt: str
                The system prompt to use for the LLM
            procedures: Dict[str, CatTool | CatForm]
                Dictionary of procedures the LLM can use
            chat_history: List[CatMessage | HumanMessage] | None
                The chat history to use for the LLM
            max_procedures_calls: int
                The maximum number of procedures the LLM can call
            chain_name: str
                The name of the chain for logging purposes

        Returns
        -------
            List[AgentOutput]
                List of outputs from the LLM and executed procedures
        """
        if chat_history is None:
            chat_history = cat.working_memory.history[-MAX_WORKING_HISTORY_LENGTH:]
        
        langchain_chat_history = []
        for message in chat_history:
            ml = message.langchainfy()
            if isinstance(ml, list):
                langchain_chat_history.extend(ml)
            else:
                langchain_chat_history.append(ml)
        
        # Create the prompt for the LLM
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_prompt.rstrip().lstrip()),
                *(langchain_chat_history),
            ]
        )

        if procedures:
            llm = cat._llm.bind_tools(self._to_langchain_tools(procedures.values()))
        else:
            llm = cat._llm

        # Create the chain for tool selection
        chain = (
            prompt
            | RunnableLambda(lambda x: langchain_log_prompt(x, chain_name))
            | llm
            | RunnableLambda(lambda x: langchain_log_output(x, f"{chain_name} - LLM output"))
        )
        
        # Execute the chain
        res: AIMessage = chain.invoke(
            {},
            config=RunnableConfig(callbacks=[
                NewTokenHandler(cat),
                ModelInteractionHandler(cat, get_caller_info(skip=1))
            ])
        )

        tool_calls: List[LLMAction] = []
        if len(res.tool_calls) > 0:
            valid_calls = [
                (res.tool_calls[i]["name"], res.tool_calls[i]["args"], res.tool_calls[i]['id'])
                for i in range(min(max_procedures_calls, len(res.tool_calls)))
                if res.tool_calls[i]["name"] in procedures.keys()
            ]

            for tool_call in valid_calls:
                settings = cat.mad_hatter.get_plugin().load_settings()
                if settings.get("stream_tool_calls", True):
                    # Stream tool calls to the chat
                    cat.send_ws_message(
                        f"Executing: `{tool_call[0]}`\n",
                        msg_type="chat_token"
                    )

                action_result = self.execute_procedure(cat, procedures[tool_call[0]], tool_call[1])

                # Set the id given by the LLM for the tool call
                # this is required by the LLM api to match the tool call with the result
                action_result.id = tool_call[2] 

                tool_calls.append(action_result)

        return AgentOutput(
            output=res.text() if res.text() else None,
            actions=tool_calls,
        )
            
    def _to_langchain_tools(self, procedures: List[CatTool | CatForm]) -> List[Tool]:
        """Convert Cheshire Cat procedures to LangChain tools."""
        langchain_tools: List[Tool] = []
        
        for p in procedures:
            if Plugin._is_cat_tool(p):
                if getattr(p, "arg_schema", None) is not None:
                    new_tool = StructuredTool(
                        name=p.name.strip().replace(" ", "_"),
                        description=p.description,
                        func=self.remove_cat_from_args(p.func),
                        args_schema=p.arg_schema,
                    )
                else:
                    new_tool = StructuredTool.from_function(
                        name=p.name.strip().replace(" ", "_"),
                        description=p.description,
                        func=self.remove_cat_from_args(p.func),
                    )
                langchain_tools.append(new_tool)
                
            elif Plugin._is_cat_form(p):
                langchain_tools.append(
                    Tool(
                        name=p.name.strip().replace(" ", "_"),
                        description=p.description,
                        func=None
                    )
                )
        
        # Add no_action tool
        langchain_tools.append(
            Tool(
                name="no_action",
                description="Use this action if no relevant action is available",
                func=None,
            )
        )
        
        return langchain_tools
    
    @staticmethod
    def remove_cat_from_args(function: Callable) -> Callable:
        """Remove 'cat' and '_' parameters from function signature for LangChain compatibility."""
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        
        filtered_parameters = [p for p in parameters if p.name != 'cat' and p.name != '_']
        new_signature = signature.replace(parameters=filtered_parameters)
        
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if 'cat' in kwargs:
                del kwargs['cat']
            return function(*args, **kwargs)
        
        wrapper.__signature__ = new_signature
        return wrapper
