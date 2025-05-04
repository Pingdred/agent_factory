import time
from datetime import timedelta
from typing import Dict, List, Tuple

from cat.agents import AgentOutput
from cat.looking_glass import prompts
from cat.log import log
from cat.mad_hatter.mad_hatter import MadHatter
from cat.utils import verbal_timedelta, match_prompt_variables

from .base import LangchainBaseAgent
from ..settings import BaseSettings


class MemoryAgent(LangchainBaseAgent):

    def __init__(self):
        super().__init__()
        self.mad_hatter = MadHatter()

    def execute(self, cat) -> AgentOutput:
        llm_message = self.run_chain(
            cat=cat,
            system_prompt=self._get_prompt(cat),
            chain_name="Memory Agent",
        )

        if isinstance(llm_message, str):
            return AgentOutput(output=llm_message)
        
        log.error(f"MemoryAgent: LLM returned an unexpected type: {type(llm_message)}")
        return AgentOutput()
    
    def _get_prompt(self, cat) -> str:
        # Obtain prompt parts from plugins
        settings = BaseSettings(**cat.mad_hatter.get_plugin().load_settings())
        prompt_prefix = settings.system_prompt
        if not settings.set_system_prompt:
            # Get system prompt from plugins
            prompt_prefix = self.mad_hatter.execute_hook(
                "agent_prompt_instructions", prompt_prefix, cat=cat
            )

        prompt_prefix = self.mad_hatter.execute_hook(
            "agent_prompt_prefix", prompt_prefix, cat=cat
        )
        prompt_suffix = self.mad_hatter.execute_hook(
            "agent_prompt_suffix", prompts.MAIN_PROMPT_SUFFIX, cat=cat
        )

        # Add episodic and declarative memories to the prompt
        # if thhere is any recalled
        recalled_episodic = self.get_recalled_episodic_memory(cat)
        recalled_declarative = self.get_recalled_declarative_memory(cat)
        prompt_variables = {}
        if len(recalled_episodic) > 0:
            prompt_variables["episodic_memory"] = self._format_episodic(recalled_episodic)
        if len(recalled_declarative) > 0:
            prompt_variables["declarative_memory"] = self._format_declarative(recalled_declarative)

        # Ensure prompt variables and placeholders match
        prompt_variables, prompt = match_prompt_variables(prompt_variables, prompt_prefix + prompt_suffix)
        prompt = prompt.format(**prompt_variables)
        return prompt
    
    def _format_episodic(self, memories: List[Tuple[str, Dict]]) -> str:
        episodic_texts = []
        for m in memories:
            # Get Time information in the Document metadata
            timestamp = m[1]["when"]
            # Get Current Time - Time when memory was stored
            delta = timedelta(seconds=(time.time() - timestamp))
            # Convert and Save timestamps to Verbal (e.g. "2 days ago")
            episodic_texts.append(f"{m[0]} ({verbal_timedelta(delta)})")

        # Format the memories for the output
        memories_separator = "\n  - "
        return (
            "## Context of things the Human said in the past: "
            + memories_separator
            + memories_separator.join(episodic_texts)
        )

    def _format_declarative(self, memories: List[Tuple[str, Dict]]) -> str:
        declarative_texts = []
        for m in memories:
            # Get and save the source of the memory
            source = m[1]["source"]
            declarative_texts.append(f"{m[0]} (extracted from {source})")

        # Format the memories for the output
        memories_separator = "\n  - "
        return (
            "## Context of documents containing relevant information: "
            + memories_separator
            + memories_separator.join(declarative_texts)
        )
