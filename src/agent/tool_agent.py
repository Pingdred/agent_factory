import random
from typing import Set, List, Tuple

from cat.log import log
from cat.agents.form_agent import FormAgent
from cat.agents.base_agent import AgentOutput
from cat.mad_hatter.mad_hatter import MadHatter

from .base import LangchainBaseAgent
from ..convo.messages import LLMAction

BASE_TOOL_PROMPT = "You are a tool agent. You can use the following tools to help the user fulfill their request."

class NativeToolAgent(LangchainBaseAgent):

    def __init__(self):
        super().__init__()
        self.mad_hatter = MadHatter()
        self.form_agent = FormAgent()

    def execute(self, cat) -> List[Tuple[LLMAction, AgentOutput]]:
        recalled_procedures_names = self.get_recalled_procedures_names(cat)
        log.debug(f"Recalled procedures: {recalled_procedures_names}")
        
        actions = self._choose_procedure(cat, recalled_procedures_names)
        if len(actions) == 0:
            return []
        
        results = []
        for a in actions:
            procedure_result = self.execute_action(
                cat, 
                action=a,
            )
            results.append((a, procedure_result))

        return results

    def _choose_procedure(self, cat, procedures_names: Set[str]) -> List[LLMAction]:
        system_prompt = self.mad_hatter.execute_hook(
            "agent_prompt_instructions", BASE_TOOL_PROMPT, cat=cat
        )

        # Gather recalled procedures
        procedures_names = self.mad_hatter.execute_hook(
            "agent_allowed_tools", procedures_names, cat=cat
        )

        if len(procedures_names) == 0:
            log.debug("No procedures available")
            return []
        
        log.debug(f"Allowed procedures: {procedures_names}")

        # Get the procedures objects (CatTool or CatForm)
        # from the recalled procedures names
        procedures = self.get_procedures(procedures_names)

        # Add procedure examples to the system prompt
        system_prompt += "\nHere some examples:\n"
        system_prompt += '\n'.join(self.generate_examples(procedures))

        SettingsModel: type = cat.mad_hatter.get_plugin().settings_model()
        settings = SettingsModel(**cat.mad_hatter.get_plugin().load_settings())

        calls = self.run_chain(
            cat=cat,
            system_prompt=system_prompt,
            procedures=procedures.values(),
            max_procedures_calls=settings.max_tools_call,
            chain_name="Tool Selection",
        )
        # No procedure selected, discard the result
        # and exit the agent
        if isinstance(calls, str):
            log.debug("No tool selected.")
            return []
        
        # Filter the calls to keep only the calls to 
        # the available procedures during this call
        valid_calls = [call for call in calls if call.name in procedures_names]

        if len(valid_calls) == 0:
            log.debug("No valid tools selected.")

        return valid_calls
     
    def generate_examples(self, allowed_procedures):
        examples = []
        for proc in allowed_procedures.values():
            if not proc.start_examples:
                continue

            example = f"user query: {random.choice(proc.start_examples)}\n"
            example += f"tool name: {proc.name}\n"

            examples.append(example)

        return examples    