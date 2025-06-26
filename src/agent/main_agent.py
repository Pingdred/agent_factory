from cat.agents import AgentOutput
from cat.agents.form_agent import FormAgent
from cat.mad_hatter.mad_hatter import MadHatter
from cat.convo.messages import CatMessage
from cat.log import log

from .base import NewBaseAgent
from .tool_agent import NativeToolAgent
from .memory_agent import MemoryAgent


class MainAgent(NewBaseAgent):

    def __init__(self):
        super().__init__()
        self.mad_hatter = MadHatter()
        self.form_agent = FormAgent()
        self.procedure_agent = NativeToolAgent()
        self.memory_agent = MemoryAgent()

    def execute(self, cat) -> AgentOutput:
        """Execute the agents.

        Returns
        -------
        agent_output : AgentOutput
            Reply of the agent, instance of AgentOutput.
        """
        log.warning(
            "The result of the hook `before_agent_starts` is not used in the agent."
            "The cat.working_memory.agent_input is not used in this agent."
        )
        self.mad_hatter.execute_hook("before_agent_starts", {}, cat=cat)

        if fast_reply := self._execute_fast_reply(cat):
            return fast_reply

        # If there is an active form, give the control to the form agent
        # to handle the form filling process
        if cat.working_memory.active_form is not None:
            form_output = self.form_agent.execute(cat)
            if form_output.return_direct:
                return form_output

        # Run tools and and start forms
        procedures_result = self.procedure_agent.execute(cat)
        if len(procedures_result) == 1 and procedures_result[0][1].return_direct:
            return procedures_result[0][1]

        intermediate_steps = []
        for action, result in procedures_result:
            if result.return_direct:
                why = cat._StrayCat__build_why()
                why.intermediate_steps = result.intermediate_steps

                cat.send_chat_message(CatMessage(
                    user_id=cat.user_id,
                    text=result.output,
                    why=why
                ))
                continue
            
            # Save the action result in the chat history
            # to be used in the memory agent
            if result.output is not None:
                self.save_action_result(action, result, cat)

            if result.intermediate_steps is not None:
                intermediate_steps += result.intermediate_steps

        memory_agent_out = self.memory_agent.execute(cat)
        memory_agent_out.intermediate_steps += intermediate_steps

        return memory_agent_out
    
    def _execute_fast_reply(self, cat) -> AgentOutput | None:
        agent_fast_reply = self.mad_hatter.execute_hook(
            "agent_fast_reply", {}, cat=cat
        )
        if isinstance(agent_fast_reply, AgentOutput):
            return agent_fast_reply
        if isinstance(agent_fast_reply, dict) and "output" in agent_fast_reply:
            return AgentOutput(**agent_fast_reply)
        
        return None

    def _execute_active_form(self, cat) -> AgentOutput | None:
        """Execute the active form.

        If there is an active form, give the control to the form agent
        to handle the form filling process.
        """
        if cat.working_memory.active_form is not None:
            form_output = self.form_agent.execute(cat)
            if form_output.return_direct:
                return form_output

        return None
