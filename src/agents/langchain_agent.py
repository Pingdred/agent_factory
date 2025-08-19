"""
Agent Factory - LangChain Agent

This module contains the LangChain-based agent implementation with function calling 
capabilities and advanced prompt management.

Classes
-------
LangchainBaseAgent
    LangChain-based agent with function calling capabilities and advanced prompt management.
"""

import inspect
import functools
from typing import Dict, List, Callable

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from langchain_core.tools import StructuredTool, Tool
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm
from cat.looking_glass.stray_cat import StrayCat
from cat.looking_glass.callbacks import ModelInteractionHandler, NewTokenHandler
from cat.memory.working_memory import MAX_WORKING_HISTORY_LENGTH
from cat.utils import get_caller_info, langchain_log_output, langchain_log_prompt
from cat.convo.messages import CatMessage, HumanMessage, ConversationMessage

from .base_agent import BaseAgent
from ..messages import LLMAction, AgentOutput


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
        execute_procedures: bool = True,
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
            execute_procedures: bool
                Whether to execute the procedures or just return the calls
            chain_name: str
                The name of the chain for logging purposes

        Returns
        -------
            AgentOutput
                Output from the LLM and executed procedures
        """
        # Prepare chat history
        langchain_chat_history = self._prepare_chat_history(cat, chat_history)

        # Create prompt template
        prompt = self._create_prompt_template(system_prompt, langchain_chat_history)

        # Prepare LLM with tools if available
        llm = cat._llm
        if procedures:
            langchain_tools = self._to_langchain_tools(procedures.values())
            llm = cat._llm.bind_tools(langchain_tools)

        # Create and execute the chain
        res = self._execute_prompt(
            prompt=prompt, llm=llm, chain_name=chain_name, cat=cat
        )

        procedures = {k.strip().replace(" ", "_"): v for k,v in procedures.items()}

        # Filter valid tool calls
        valid_calls = [
            LLMAction(
                id=res.tool_calls[i]["id"],
                name=res.tool_calls[i]["name"],
                input=res.tool_calls[i]["args"],
                return_direct=getattr(procedures[res.tool_calls[i]["name"]], "return_direct", True),  # Default to True if not set
            )
            for i in range(min(max_procedures_calls, len(res.tool_calls)))
            if res.tool_calls[i]["name"] in procedures.keys()
        ]

        # Execute procedures if requested
        if execute_procedures:
            valid_calls = [
                self.execute_action(tool_call, cat) for tool_call in valid_calls
            ]

        # Ensure the llm message is a non-empty string or None
        text_output = res.content if res.content else None

        return AgentOutput(
            output=text_output,
            actions=valid_calls,
        )

    def _prepare_chat_history(self, cat: StrayCat, chat_history: List[ConversationMessage] | None = None) -> List[LangchainBaseMessage]:
        """
        Prepare chat history for LangChain processing.
        
        This method converts the chat history from Cheshire Cat format to
        LangChain's message format. It ensures that the chat history is
        limited to the last MAX_WORKING_HISTORY_LENGTH messages if no specific
        chat history is provided.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for the chat history.
        chat_history : List[ConversationMessage] | None, optional
            The chat history to convert. If None, the last MAX_WORKING_HISTORY_LENGTH
            messages from the working memory will be used.

        Returns
        -------
        List[LangchainBaseMessage]
            A list of LangChain messages (SystemMessage, AIMessage) representing
            the chat history.
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
        
        return langchain_chat_history

    def _create_prompt_template(self, system_prompt: str, langchain_chat_history: List[LangchainBaseMessage]) -> ChatPromptTemplate:
        """      
        This method constructs a LangChain ChatPromptTemplate using the provided
        system prompt and chat history. It ensures the system prompt is properly
        formatted by stripping leading and trailing whitespace.

        Parameters
        ----------
        system_prompt : str
            The system prompt to use for the LLM. It should be a well-formed string
            that provides context or instructions for the LLM.
        langchain_chat_history : List[LangchainBaseMessage]
            The chat history to include in the prompt. This should be a list of
            LangChain messages (e.g., SystemMessage, AIMessage) that represent the
            conversation history leading up to the current interaction.
        
        Returns
        -------
        ChatPromptTemplate
            A LangChain ChatPromptTemplate object that combines the system prompt
            and the chat history.
        """
        return ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_prompt.rstrip().lstrip()),
                *langchain_chat_history,
            ]
        )

    def _execute_prompt(self, prompt: ChatPromptTemplate, llm, chain_name: str, cat: StrayCat) -> AIMessage:
        """
        Execute a prompt using a LangChain chain with logging and Cheshire Cat callbacks.
        
        This method creates a LangChain chain that processes the prompt,
        logs the prompt and output, and invokes the LLM with the provided
        callbacks for token handling and metadata logging.

        Parameters
        ----------
        prompt : ChatPromptTemplate
            The prompt template to use for the LLM.
        llm : LLM
            The LangChain LLM instance to use for generating responses.
        chain_name : str
            The name of the chain for logging purposes.
        cat : StrayCat
            The StrayCat instance providing context for the execution.  

        Returns
        -------
        AIMessage
            The AIMessage containing the LLM's response after processing the prompt.
        """
        chain = (
            prompt
            | RunnableLambda(lambda x: langchain_log_prompt(x, chain_name))
            | llm
            | RunnableLambda(lambda x: langchain_log_output(x, f"{chain_name} - LLM output"))
        )

        return chain.invoke(
            {},
            config=RunnableConfig(callbacks=[
                NewTokenHandler(cat),
                ModelInteractionHandler(cat, get_caller_info(skip=1))
            ])
        )

    def _to_langchain_tools(self, procedures: List[CatTool | CatForm]) -> List[Tool]:
        """
        Convert Cheshire Cat procedures to LangChain tools.
        
        This method iterates through the provided procedures and converts
        each CatTool and CatForm into a LangChain-compatible Tool or StructuredTool.

        Parameters
        ----------
        procedures : List[CatTool | CatForm]
            List of procedures (CatTool or CatForm) to convert.

        Returns
        -------
        List[Tool]
            A list of LangChain tools created from the provided procedures.
        """
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
        """
        Remove 'cat' and '_' parameters from function signature for LangChain compatibility.
        
        Parameters
        ----------
        function : Callable
            The function to modify.

        Returns
        -------
        Callable
            The modified function without 'cat' and '_' parameters.
        """
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
