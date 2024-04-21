import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message
from .base import IntelligenceBackend

try:
    # from langchain.llms import OpenAI
    from langchain_community.llms import Ollama
except ImportError:
    is_langchain_openai_available = False
    # logging.warning("openai package is not installed")
is_langchain_openai_available = True

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "llama3:instruct"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|reserved_special_token",
    END_OF_MESSAGE)  # End of sentence token
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."


class LangChainOpenAIChat2(IntelligenceBackend):
    """Interface to the ChatGPT style model with system, user, assistant roles separation."""

    stateful = False
    type_name = "openai-chat"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        merge_other_agents_as_one_user: bool = True,
        **kwargs,
    ):
        """
        Instantiate the OpenAIChat backend.

        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        assert (
            is_langchain_openai_available
        ), "langchain package is not installed or the API key is not set"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user
        self.llm = Ollama(
            model=model,
            temperature=temperature,
            # max_tokens=max_tokens,
            stop=STOP,
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        response = self.llm(prompt=messages)
        return response

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Format the input and call the ChatGPT/GPT-4 API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """

        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name: {agent_name}\n\nYour role:{role_desc}"
        else:
            system_prompt = (
                f"You are {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"
            )

        all_messages = [(SYSTEM_NAME, system_prompt)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:  # non-system messages are suffixed with the end of message token
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))

        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content))
        else:  # The default request message that reminds the agent its role and instruct it to speak
            all_messages.append(
                (SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}")
            )

        messages = []
        for i, msg in enumerate(all_messages):
            if i == 0:
                assert (
                    msg[0] == SYSTEM_NAME
                )  # The first message should be from the system
                messages.append({"role": "system", "content": msg[1]})
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    if messages[-1]["role"] == "user":  # last message is from user
                        if self.merge_other_agent_as_user:
                            messages[-1][
                                "content"
                            ] = f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
                        else:
                            messages.append(
                                {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                            )
                    elif (
                        messages[-1]["role"] == "assistant"
                    ):  # consecutive assistant messages
                        # Merge the assistant messages
                        messages[-1]["content"] = f"{messages[-1]['content']}\n{msg[1]}"
                    elif messages[-1]["role"] == "system":
                        messages.append(
                            {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                        )
                    else:
                        raise ValueError(f"Invalid role: {messages[-1]['role']}")

 
        response = self._get_response(messages, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response
        ).strip()  # noqa: F541

        # Remove the tailing end of message token
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response

class LangChainOpenAIChat(IntelligenceBackend):
    """Interface to the Cohere API."""

    stateful = True
    type_name = "cohere-chat"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature, max_tokens=max_tokens, model=model, **kwargs
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        assert (
            is_langchain_openai_available
        ), "langchain package is not installed or the API key is not set"

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        # self.merge_other_agent_as_user = merge_other_agents_as_one_user
        self.llm = Ollama(
            model=model,
            temperature=temperature,
            # max_tokens=max_tokens,
            stop=STOP,
        )

        # Stateful variables
        self.session_id = None  # The session id for the last conversation
        self.last_msg_hash = (
            None  # The hash of the last message of the last conversation
        )

    def reset(self):
        self.session_id = None
        self.last_msg_hash = None

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, new_message: str):#, persona_prompt: str):
        response = self.llm(
            new_message,
            # persona_prompt=persona_prompt,
            # temperature=self.temperature,
            # max_tokens=self.max_tokens,
            # session_id=self.session_id,
        )

        # self.session_id = response.session_id  # Update the session id
        return response

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Format the input and call the Cohere API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request for the CohereAI
        """
        # Find the index of the last message of the last conversation
        new_message_start_idx = 0
        if self.last_msg_hash is not None:
            for i, message in enumerate(history_messages):
                if message.msg_hash == self.last_msg_hash:
                    new_message_start_idx = i + 1
                    break

        new_messages = history_messages[new_message_start_idx:]
        assert len(new_messages) > 0, "No new messages found (this should not happen)"

        new_conversations = []
        for message in new_messages:
            if message.agent_name != agent_name:
                # Since there are more than one player, we need to distinguish between the players
                new_conversations.append(f"[{message.agent_name}]: {message.content}")

        if request_msg:
            new_conversations.append(
                f"[{request_msg.agent_name}]: {request_msg.content}"
            )

        # Concatenate all new messages into one message because the Cohere API only accepts one message
        new_message = "\n".join(new_conversations)
        persona_prompt = f"System:\n{global_prompt}\n\nYour role:\n{role_desc}"

        response = self._get_response(persona_prompt + "\n\n" + new_message)

        # Only update the last message hash if the API call is successful
        # self.last_msg_hash = new_messages[-1].msg_hash

        return response
