"""
Agent Factory - Agent Classes

This module contains all the agent classes for the Agent Factory plugin.
It includes the base agent functionality and the LangChain-based agent implementation.

Classes
-------
BaseAgent
    Base agent class that extends Cheshire Cat's BaseAgent with enhanced functionality.
LangchainBaseAgent
    LangChain-based agent with function calling capabilities and advanced prompt management.

Functions
---------
_execute_tool(cat, tool, input)
    Execute a CatTool with the provided input and return the result.
_execute_form(cat, form)
    Execute a CatForm and return the result.
"""

import inspect
import functools
from typing import Dict, List, Set, Tuple, Callable, Any

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from langchain_core.tools import StructuredTool, Tool
from langchain_core.runnables import RunnableConfig, RunnableLambda

from cat.mad_hatter.mad_hatter import MadHatter
from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm, CatFormState
from cat.agents.base_agent import BaseAgent as CatBaseAgent
from cat.agents.form_agent import FormAgent
from cat.looking_glass.stray_cat import StrayCat
from cat.looking_glass.callbacks import ModelInteractionHandler, NewTokenHandler
from cat.memory.working_memory import MAX_WORKING_HISTORY_LENGTH
from cat.utils import get_caller_info, langchain_log_output, langchain_log_prompt
from cat.convo.messages import CatMessage, HumanMessage, ConversationMessage
from cat.log import log

from .messages import LLMAction, CatToolMessage, AgentOutput


# Agent execution utilities
def _execute_tool(cat: StrayCat, tool: CatTool, input: Dict[str, Any]) -> LLMAction:
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

def _execute_form(cat: StrayCat, form: CatForm) -> LLMAction:
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


class BaseAgent(CatBaseAgent):
    """
    Base agent class that extends Cheshire Cat's BaseAgent.
    
    This class provides common functionality for all custom agents including
    procedure execution, memory access, and result saving.
    """

    def __init__(self):
        super().__init__()

    def execute(self, cat: StrayCat) -> AgentOutput:
        """
        Execute the agent logic.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for execution.

        Returns
        -------
        AgentOutput
            The output of the agent execution.
        """
        pass

    @staticmethod
    def execute_procedure(procedure: CatTool | CatForm, input: str | Dict, cat: StrayCat, call_id: str| None = None) -> LLMAction:
        """
        Execute a procedure (tool or form) with the given input.

        Parameters
        ----------
        procedure : CatTool | CatForm
            The procedure to execute (either a tool or form).
        input : str | Dict
            The input data to pass to the procedure.
        cat : StrayCat
            The StrayCat instance providing context for execution.
        call_id : str | None, optional
            The call ID to preserve for LLM API matching, by default None.

        Returns
        -------
        LLMAction
            The result of procedure execution.

        Raises
        ------
        ValueError
            If the procedure type is not recognized.
        """

        settings = cat.mad_hatter.get_plugin().load_settings()
        if settings.get("stream_tool_calls", True):
            # Stream tool calls to the chat
            cat.send_ws_message(
                f"Executing: `{procedure.name}`\n",
                msg_type="chat_token"
            )

        if settings.get("notify_tool_calls", False):
            # Notify the user about the tool call
            cat.send_notification(f"Executing: `{procedure.name}`")

        try:
            if Plugin._is_cat_tool(procedure):
                res = _execute_tool(cat, procedure, input)
                res.id = call_id  # Preserve the original call ID, required by the LLM api to match the tool call with the result
                return res
            
            if Plugin._is_cat_form(procedure):
                res = _execute_form(cat, procedure)
                res.id = call_id  # Preserve the original call ID, required by the LLM api to match the tool call with the result
                return res
            
        except Exception as e:
            log.error(f"Error executing {procedure.procedure_type} `{procedure.name}`: {str(e)}")

        log.error(f"Unknown procedure type: {type(procedure)}")
        raise ValueError(
            f"Unknown procedure type: {type(procedure)}. "
            "Expected CatTool or CatForm."
        )
    
    def execute_action(self, action: LLMAction, cat: StrayCat) -> LLMAction:
        """
        Execute an action by finding and running the corresponding procedure.
        
        Parameters
        ----------
        action : LLMAction
            The action to execute, containing the procedure name and input.
        cat : StrayCat
            The StrayCat instance providing context for execution.
        
        Returns
        -------
        LLMAction
            The result of the executed action, including output and metadata.

        Raises
        ------
        ValueError
            If the action's procedure is not found in the MadHatter registry.
        """
        procedures = self.get_procedures([action.name])

        if not procedures:
            log.error(f"Action {action.name} not found.")
            raise ValueError(f"Action {action.name} not found.")
        
        # Select the firsr and only procedure from the dictionary
        selected_procedure = list(procedures.values())[0]
        
        return self.execute_procedure(selected_procedure, action.input, cat, call_id=action.id)
    
    def save_action(self, action: LLMAction, cat: StrayCat) -> None:
        """
        Save the action result in the chat history.

        Parameters
        ----------
        action : LLMAction
            The action to save, containing the procedure name, input, and output.
        cat : StrayCat
            The StrayCat instance providing context for saving the action.

        Notes
        -----
        This method creates a CatToolMessage with the action details and
        updates the working memory history with the action call.
        """
        action_call = CatToolMessage(
            user_id=cat.user_id,
            action=action,
        )
        cat.working_memory.update_history(action_call)

    def get_recalled_procedures(self, cat: StrayCat) -> Dict[str, CatTool | CatForm]:
        """
        Get the recalled Tools and Forms from the working memory.
        This method retrieves the names of procedures recalled from the
        working memory and fetches their definitions from the MadHatter.
        
        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for procedure retrieval.
        
        Returns
        -------
        Dict[str, CatTool | CatForm]
            A dictionary mapping procedure names to their corresponding
            CatTool or CatForm objects. If no procedures are recalled,
            an empty dictionary is returned.

        Notes
        -----
        The keys in the returned dictionary are normalized to valid identifiers
        by stripping whitespace and replacing spaces with underscores, as required
        by LLM APIs. This ensures compatibility with the tool calling mechanism.
        """
        recalled_procedures_names = self.get_recalled_procedures_names(cat)
        if not recalled_procedures_names:
            return {}

        # Get the procedures by name from the MadHatter
        procedures = self.get_procedures(list(recalled_procedures_names))
        
        if not procedures:
            log.warning("No procedures found in the MadHatter for recalled names.")
        
        return procedures

    def get_recalled_procedures_names(self, cat: StrayCat) -> Set[str]:
        """
        Get the names of the recalled Tools and Forms from the working memory.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for procedure retrieval.

        Returns
        -------
        Set[str]
            A set of names of recalled procedures (Tools and Forms) from the
            working memory. If no procedures are recalled, an empty set is returned.
        """
        recalled_procedures_names = set()
        
        for memory_group in cat.working_memory.procedural_memories:
            memory = memory_group[0]
            metadata = memory.metadata
            
            if (metadata["type"] in ["tool", "form"] and 
                metadata["trigger_type"] in ["description", "start_example"]):
                recalled_procedures_names.add(metadata["source"])
                
        return recalled_procedures_names
    
    def get_recalled_episodic_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """
        Get the recalled episodic memory from the working memory.

        This method retrieves the episodic memories stored in the working memory
        and returns them as a list of tuples containing the text content and metadata.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for memory retrieval.

        Returns
        -------
        List[Tuple[str, Dict]]
            A list of tuples where each tuple contains the stripped text content
            and metadata from the episodic memories. If no episodic memories are found,
            an empty list is returned.
        """
        return self._memory_points_to_tuple(cat.working_memory.episodic_memories)
    
    def get_recalled_declarative_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """
        Get the recalled declarative memory from the working memory.
        
        This method retrieves the declarative memories stored in the working memory
        and returns them as a list of tuples containing the text content and metadata.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance providing context for memory retrieval.
        
        Returns
        -------
        List[Tuple[str, Dict]]
            A list of tuples where each tuple contains the stripped text content
            and metadata from the declarative memories. If no declarative memories are found,
            an empty list is returned.
        """
        return self._memory_points_to_tuple(cat.working_memory.declarative_memories)
    
    def handle_active_form(self,cat: StrayCat) -> AgentOutput | None:
        """
        Handle active form processing if one exists in working memory.

        This method checks for an active form in the working memory and
        processes it accordingly. It handles form closure, continuation,
        and error scenarios.

        Parameters
        ----------
        cat : StrayCat
            The StrayCat instance whose working memory may contain an active form.

        Returns
        -------
        AgentOutput | None
            AgentOutput containing form results if form processing succeeds,
            None if no active form exists, form is closed, or an error occurs.

        Notes
        -----
        If a form is in CLOSED state, it will be removed from working memory.
        If form processing fails, an error is logged and None is returned.
        """
        # get active form from working memory
        active_form = cat.working_memory.active_form
        
        if not active_form:
            # no active form
            return None
        
        if active_form._state == CatFormState.CLOSED:
            # form is closed, delete it from working memory
            cat.working_memory.active_form = None
            return None
        
        # continue form
        try:
            form_output = active_form.next()
            return AgentOutput(
                output=form_output["output"],
                actions=[LLMAction(
                    action=active_form.name,
                    action_input=active_form._model
                )]
            )
        except Exception:
            log.error("Error while executing form")
            return None

    def _memory_points_to_tuple(self, memory_points: List[Tuple[Document, Dict]]) -> List[Tuple[str, Dict]]:
        """
        Convert memory points to tuples of (text, metadata).

        This helper method extracts the page content and metadata from
        Document objects in memory points and returns them as simple tuples.

        Parameters
        ----------
        memory_points : List[Tuple[Document, Dict]]
            List of memory point tuples containing Document objects and their metadata.

        Returns
        -------
        List[Tuple[str, Dict]]
            List of tuples containing (stripped_text_content, metadata) from
            each Document in the memory points.
        """
        return [
            (m[0].page_content.strip(), m[0].metadata)
            for m in memory_points
        ]

    def get_procedures(self, procedures_name: List[str]) -> Dict[str, CatTool | CatForm]:
        """
        Get procedures by name from the MadHatter.

        This method retrieves procedures from the MadHatter registry based on
        the provided names. It normalizes procedure names by replacing spaces
        with underscores to ensure compatibility with LLM API requirements.

        Parameters
        ----------
        procedures_name : List[str]
            List of procedure names to retrieve. Names will be normalized
            by stripping whitespace and replacing spaces with underscores.

        Returns
        -------
        Dict[str, CatTool | CatForm]
            Dictionary mapping normalized procedure names to their corresponding
            procedure objects. Only procedures found in MadHatter are included.

        Notes
        -----
        Procedure names are normalized to valid identifiers by removing leading/
        trailing whitespace and replacing spaces with underscores, as required
        by LLM APIs.
        """
        # Clean up procedure names to ensure they are valid identifiers
        # LLM's API require tool names without spaces and special characters
        procedures_name = [p.strip().replace(" ", "_") for p in procedures_name]

        allowed_procedures: Dict[str, CatTool | CatForm] = {}
        for procedure in MadHatter().procedures:
            # Ensure the procedure name is a valid identifier
            # LLM's api require tool names without spaces and special characters
            p_name = procedure.name.strip().replace(" ", "_")

            if p_name in procedures_name:                
                allowed_procedures[p_name] = procedure
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
        text_output = res.text() if res.text() else None

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
