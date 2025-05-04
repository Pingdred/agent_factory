import inspect
import functools
from typing import Dict, List, Set, Callable, Any, Tuple

from cat.log import log
from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm
from cat.convo.messages import CatMessage, HumanMessage
from cat.agents.base_agent import AgentOutput, BaseAgent
from cat.agents.form_agent import FormAgent
from cat.looking_glass.callbacks import ModelInteractionHandler
from cat.memory.working_memory import MAX_WORKING_HISTORY_LENGTH
from cat.utils import get_caller_info, langchain_log_output, langchain_log_prompt

from ..convo.messages import LLMAction, CatToolMessage


def _execute_procedure(cat, procedure: CatTool | CatForm, input: str | Dict) -> AgentOutput:
    # NOTE: This function can be moved to the MadHatter
    try:
        if Plugin._is_cat_tool(procedure):
            return _execute_tool(cat, procedure, input)
        
        if Plugin._is_cat_form(procedure):
            return _execute_form(cat, procedure)
        
    except Exception as e:
        log.error(f"Error executing {procedure.procedure_type} `{procedure.name}`: {str(e)}")

    log.error(f"Unknown procedure type: {type(procedure)}")
    return AgentOutput()
  
def _execute_tool(cat, tool: CatTool, input: Dict[str, Any]) -> AgentOutput:
    log.debug(f"Executing tool: {tool.name} with input: {input}")
    tool_output = tool.func(
        **input, cat=cat
    )

    # Ensure the output is a string or None, 
    if (tool_output is not None) and (not isinstance(tool_output, str)):
        tool_output = str(tool_output)

    return AgentOutput(
        output=tool_output,
        return_direct=tool.return_direct,
        intermediate_steps=[
            ((tool.procedure_type, tool.name), tool_output)
        ]
    )

def _execute_form(cat, form: CatForm) -> AgentOutput:
    form_instance = form(cat)
    cat.working_memory.active_form = form_instance
    return FormAgent().execute(cat)


class NewBaseAgent(BaseAgent):

    def __init__(self):
        super().__init__()

    def execute_procedure(self, cat, name: str, input: str | Dict) -> AgentOutput:
        procedure = self._get_procedures([name])
        if procedure:
            procedure = procedure[name]
            return _execute_procedure(cat, procedure, input)
        
        raise ValueError(f"Procedure `{name}` not found in the available procedures.")

    def save_action_result(self, action: LLMAction, procedure_result: AgentOutput, cat) -> None:
        # Save action call and result in the chat history
        # to be used in later interactions.
        action_call = CatToolMessage(
            user_id=cat.user_id,
            action=action,
            result=procedure_result,
        )
        cat.working_memory.update_history(action_call)

    def get_recalled_procedures_names(self, cat) -> Set[str]:
        recalled_procedures_names = set()
        
        for memory_group in cat.working_memory.procedural_memories:
            memory = memory_group[0]  # Get the first memory in the group
            metadata = memory.metadata
            
            # Check if this is a tool or form with appropriate trigger type
            if (metadata["type"] in ["tool", "form"] and 
                metadata["trigger_type"] in ["description", "start_example"]):
                recalled_procedures_names.add(metadata["source"])
                
        return recalled_procedures_names
    
    def get_recalled_episodic_memory(self, cat) -> List[Tuple[str, Dict]]:       
        return self._memory_points_to_tuple(cat.working_memory.episodic_memories)
    
    def get_recalled_declarative_memory(self, cat) -> List[Tuple[str, Dict]]:       
        return self._memory_points_to_tuple(cat.working_memory.declarative_memories)
    
    def _memory_points_to_tuple(self, memory_points: List[Tuple[Document, ]]) -> List[Tuple[str, Dict]]:
        # Convert memory points to a tuple of (text, metadata)
        # to be used in the prompt, metadata can be used to
        # filter memories or enrich the prompt
        return [
            (m[0].page_content.rstrip().lstrip(), m[0].metadata)
            for m in memory_points
        ]

    def _get_procedures(self, procedures_name: List[str]) -> Dict[str, CatTool | CatForm]:
        # NOTE: This function should be in the MadHatter, or much better
        # make mad_hatter.procedures a dict with the procedure name as key
        # and the procedure as value
        
        allowed_procedures: Dict[str, CatTool | CatForm] = {}
        # Filter available procedures to only include those that were recalled
        for procedure in MadHatter().procedures:
            if procedure.name in procedures_name:
                allowed_procedures[procedure.name] = procedure

        return allowed_procedures
