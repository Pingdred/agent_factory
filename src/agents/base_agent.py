"""
Agent Factory - Base Agent

This module contains the base agent class that extends Cheshire Cat's BaseAgent
with enhanced functionality for procedure execution, memory access, and result saving.

Classes
-------
BaseAgent
    Base agent class that extends Cheshire Cat's BaseAgent with enhanced functionality.
"""

from typing import Dict, List, Set, Tuple

from langchain_core.documents import Document

from cat.mad_hatter.mad_hatter import MadHatter
from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm, CatFormState
from cat.agents.base_agent import BaseAgent as CatBaseAgent
from cat.looking_glass.stray_cat import StrayCat
from cat.log import log

from ..messages import LLMAction, CatToolMessage, AgentOutput
from ..utils import execute_tool, execute_form


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
                res = execute_tool(cat, procedure, input)
                res.id = call_id  # Preserve the original call ID, required by the LLM api to match the tool call with the result
                return res
            
            if Plugin._is_cat_form(procedure):
                res = execute_form(cat, procedure)
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
    
    def handle_active_form(self, cat: StrayCat) -> AgentOutput | None:
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
                    name=active_form.name,
                    input=active_form._model
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
