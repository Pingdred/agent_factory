import inspect
import functools
from typing import Dict, List, Set, Callable, Any, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import StructuredTool, Tool
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.documents import Document

from cat.log import log
from cat.mad_hatter.plugin import Plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.mad_hatter.decorators.tool import CatTool
from cat.experimental.form.cat_form import CatForm
from cat.convo.messages import CatMessage, HumanMessage
from cat.agents.base_agent import AgentOutput, BaseAgent
from cat.agents.form_agent import FormAgent
from cat.looking_glass.stray_cat import StrayCat
from cat.looking_glass.callbacks import ModelInteractionHandler
from cat.memory.working_memory import MAX_WORKING_HISTORY_LENGTH
from cat.utils import get_caller_info, langchain_log_output, langchain_log_prompt

from ..convo.messages import LLMAction, CatToolMessage


def _execute_procedure(cat: StrayCat, procedure: CatTool | CatForm, input: str | Dict) -> AgentOutput:
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
  
def _execute_tool(cat: StrayCat, tool: CatTool, input: Dict[str, Any]) -> AgentOutput:
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
            ((tool.name, str(input)), tool_output)
        ]
    )

def _execute_form(cat: StrayCat, form: CatForm) -> AgentOutput:
    form_instance = form(cat)
    cat.working_memory.active_form = form_instance
    return FormAgent().execute(cat)


class NewBaseAgent(BaseAgent):

    def __init__(self):
        super().__init__()

    def execute(self, cat: StrayCat) -> AgentOutput:
        """Abstract method to be implemented by the agent."""
        pass

    def execute_action(self, cat: StrayCat, action: LLMAction) -> AgentOutput:
        """Execute a procedure by name.

        Parameters
        ----------
            cat: StrayCat
                The StrayCat instance
            action: LLMAction
                The action to execute 
        Returns
        -------
            AgentOutput
                The output of the procedure
        Raises
        ----------
            ValueError
                If the procedure is not found
        """

        procedure = self.get_procedures([action.name]).get(action.name, None)
        if procedure is not None:
            return _execute_procedure(cat, procedure, action.input)
        
        raise ValueError(f"Procedure `{action.name}` not found in the available procedures.")

    def save_action_result(self, action: LLMAction, action_result: AgentOutput, cat) -> None:
        """Save the action result in the chat history.
        This is used to keep track of the actions taken and their results.
        
        Parameters
        ----------
            action: LLMAction
                The action to save
            procedure_result: AgentOutput
                The result of the action
            cat: StrayCat
                The StrayCat instance
        """
        # Save action call and result in the chat history
        # to be used in later interactions.
        action_call = CatToolMessage(
            user_id=cat.user_id,
            action=action,
            result=action_result,
        )
        cat.working_memory.update_history(action_call)

    def get_recalled_procedures_names(self, cat: StrayCat) -> Set[str]:
        """Get the names of the recalled procedures from the working memory."""

        recalled_procedures_names = set()
        
        for memory_group in cat.working_memory.procedural_memories:
            memory = memory_group[0]  # Get the first memory in the group
            metadata = memory.metadata
            
            # Check if this is a tool or form with appropriate trigger type
            if (metadata["type"] in ["tool", "form"] and 
                metadata["trigger_type"] in ["description", "start_example"]):
                recalled_procedures_names.add(metadata["source"])
                
        return recalled_procedures_names
    
    def get_recalled_episodic_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """Get the recalled episodic memory from the working memory.

        Parameters
        ----------
            cat: StrayCat
                The StrayCat instance
        Returns
        -------
            List[Tuple[str, Dict]]
                A list of tuples containing the text and metadata dictionary of the episodic memories.
        """
        return self._memory_points_to_tuple(cat.working_memory.episodic_memories)
    
    def get_recalled_declarative_memory(self, cat: StrayCat) -> List[Tuple[str, Dict]]:
        """Get the recalled declarative memory from the working memory.
        
        Parameters
        ----------
            cat: StrayCat
                The StrayCat instance
        Returns
        -------
            List[Tuple[str, Dict]]
                A list of tuples containing the text and metadata dictionary of the declarative memories.
        """
        return self._memory_points_to_tuple(cat.working_memory.declarative_memories)
    
    def _memory_points_to_tuple(self, memory_points: List[Tuple[Document, ]]) -> List[Tuple[str, Dict]]:
        # Convert memory points to a tuple of (text, metadata)
        # to be used in the prompt, metadata can be used to
        # filter memories or enrich the prompt
        return [
            (m[0].page_content.strip(), m[0].metadata)
            for m in memory_points
        ]

    def get_procedures(self, procedures_name: List[str]) -> Dict[str, CatTool | CatForm]:
        # NOTE: This function should be in the MadHatter, or much better
        # make mad_hatter.procedures a dict with the procedure name as key
        # and the procedure as value
        
        allowed_procedures: Dict[str, CatTool | CatForm] = {}
        # Filter available procedures to only include those that were recalled
        for procedure in MadHatter().procedures:
            if procedure.name in procedures_name:
                allowed_procedures[procedure.name] = procedure

        return allowed_procedures
    

class LangchainBaseAgent(NewBaseAgent):

    def run_chain(
        self,
        cat: StrayCat,
        system_prompt: str,
        procedures: List[CatTool | CatForm] = [],
        chat_history: List[CatMessage | HumanMessage] | None = None,
        max_procedures_calls: int = 1,
        chain_name: str = "Langchain Chain",
    ) -> str | List[LLMAction]:
        """
        Interrogate the LLM to get a text response or a list of actions to execute.

        Parameters
        ----------
            cat: StrayCat
                The StrayCat instance
            system_prompt: str
                The system prompt to use for the LLM
            procedures: List[CatTool | CatForm]
                List of procedures the LLM can use
            chat_history: List[CatMessage | HumanMessage] | None
                The chat history to use for the LLM, if None, the last
                MAX_WORKING_HISTORY_LENGTH messages will be used
            max_procedures_calls: int
                The maximum number of procedures the LLM can call in a single interaction
            chain_name: str
                The name of the chain to use for logging purposes

        Returns
        -------
            str | List[LLMAction]
                The LLM response or a list of actions (tools or forms) to execute
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
                # Add chat history to provide context
                *(langchain_chat_history),
            ]
        )

        if procedures is not None and len(procedures) > 0:
            # Add tool to the prompt
            llm = cat._llm.bind_tools(self._to_langchain_tools(procedures))  # parallel_tool_calls=False
        else:
            llm = cat._llm

        # Create the chain for tool selection
        chain = (
            prompt
            | RunnableLambda(lambda x: langchain_log_prompt(x, chain_name))
            | llm
            | RunnableLambda(lambda x: langchain_log_output(x, f"{chain_name} - LLM output"))
        )
        
        # Execute the chain to get the LLM's action choice
        res: AIMessage = chain.invoke(
            {},
            config=RunnableConfig(callbacks=[
                ModelInteractionHandler(cat, get_caller_info(skip=1))
            ])
        )

        if len(res.tool_calls) > 0:
            procedures_names = [p.name for p in procedures]

            # Filter the tool calls to only include those that are in the allowed procedures
            # and limit the number of calls to max_procedures_calls
            # This check will also exclude the "no_action" tool added in the _to_langchain_tools method
            valid_calls = [
                LLMAction(
                    name=res.tool_calls[i]["name"],
                    input=res.tool_calls[i]["args"],
                    id=res.tool_calls[i]["id"],
                )
                for i in range(min(max_procedures_calls, len(res.tool_calls)))
                if res.tool_calls[i]["name"] in procedures_names
            ]
            return valid_calls
        
        if isinstance(res.content, str):
            # If the LLM output is a string, return it directly
            return res.content
        
        raise ValueError(f"Unexpected LLM output: {res.content}")
    
    def _to_langchain_tools(self, procedures: List[CatTool | CatForm]) -> List[Tool]:
        """
        Prepare a list of allowed procedures as LangChain tools.
        
        Parameters
        ----------
            procedures: List[CatTool | CatForm]
                List of procedures to convert to LangChain tools
            
        Returns
        -------
            List[Tool]
                List of LangChain tools
        """

        langchain_tools: List[Tool] = []
        for p in procedures:
            if Plugin._is_cat_tool(p):
                # NOTE: Code required for the plugin mcp_adapter,
                # add a langchainfy method to the CatTool class
                # can resolve this issue.
                if getattr(p, "arg_schema", None) is not None:
                    new_tool = StructuredTool(
                        name=p.name,
                        description=p.description,
                        func=self.remove_cat_from_args(p.func),
                        args_schema=p.arg_schema,
                    )
                else:
                    new_tool = StructuredTool.from_function(
                            name=p.name,
                            description=p.description,
                            func=self.remove_cat_from_args(p.func),
                        )
        
                langchain_tools.append(
                   new_tool
                )
            elif Plugin._is_cat_form(p):
                # Create a structured tool for the form
                langchain_tools.append(
                    Tool(
                        name=p.name,
                        description=p.description,
                        func=None
                    )
                )
        
        # Fake tool to give the LLM an explicit option
        # to not take any action, this is useful when the LLM
        # is not sure about the action to take, or when any 
        # of the actions are relevant.
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
        Prepare a function to be used as a LangChain tool by removing 'cat' and '_' from the signature.

        Parameters
        ----------
            function: Callable
                The function to modify

        Returns
        -------
            Callable
                The modified function with 'cat' and '_' removed from the signature
        """

        # Get the current signature
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        
        # Remove the 'cat' and '_' parameters
        # NOTE: '_' is used as a placeholder for unused parameters, leving it will cause issues
        # with the signature analysis of the StructuredTool
        filtered_parameters = [p for p in parameters if p.name != 'cat' and p.name != '_']
        
        # Create a new signature
        new_signature = signature.replace(parameters=filtered_parameters)
        
        # Create a wrapper function that doesn't accept 'cat'
        # functools.wraps is used to preserve the original function's metadata
        # and signature, this is important because the funcion info is used
        # by the StructuredTool to generate the args_schema
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Remove 'cat' from kwargs if present
            if 'cat' in kwargs:
                del kwargs['cat']
            return function(*args, **kwargs)
        
        # Apply the new signature
        wrapper.__signature__ = new_signature
        return wrapper