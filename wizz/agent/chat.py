from collections import deque

from openai import OpenAI
from openai.types.chat import ChatCompletion

from wizz.interface.enums import MessageRole
from wizz.interface.types import MessageTuple

_DEFAULT_TOKEN_LIMIT = 256


class Chat:
    """A model of a chatbot with a chat queue and a conversation history."""

    def __init__(  # noqa: WPS211
        self,
        system_message: str | None = None,
        *,
        chat_window_length: int = 10,
        model: str | None = None,
        temperature: float | None = None,
        token_limit: int | None = None,
    ):
        """Set up the chatbot with an optional system message."""
        self.prepended_messages: list[MessageTuple] = (
            [(MessageRole.system, system_message)]
            if system_message
            else []
        )
        self.appended_messages: list[MessageTuple] = []
        self.openai = OpenAI().chat.completions
        self.chat_queue: deque[MessageTuple] = deque(maxlen=chat_window_length)
        self._model = model or 'gpt-3.5-turbo'
        self._temperature = temperature or None
        self._token_limit = token_limit or _DEFAULT_TOKEN_LIMIT

    def set_prepend_messages(self, *messages: MessageTuple) -> None:
        """Set messages to be prepended to the chat queue each time."""
        self.prepended_messages = list(messages)

    def set_append_messages(self, *messages: MessageTuple) -> None:
        """Set messages to be appended to the chat queue each time."""
        self.appended_messages = list(messages)

    def ask(  # noqa: WPS211
        self,
        *messages: tuple[MessageRole, str],
        store_in_chat_queue: bool = True,
        token_limit_override: int | None = None,
        model_override: str | None = None,
        temperature_override: float | None = None,
    ) -> str:
        """Ask a question and add it to the chat queue with an answer."""
        if not messages:
            raise ValueError('At least one message is required.')
        messages_to_provide = [
            *self.prepended_messages,
            *self.chat_queue,
            *messages,
            *self.appended_messages,
        ]
        completion: ChatCompletion = self.openai.create(
            model=model_override or self._model,
            messages=[  # type: ignore
                self.to_message_dict(role, content)
                for role, content in messages_to_provide  # noqa: WPS110
            ],
            max_tokens=token_limit_override or self._token_limit,
            temperature=temperature_override or self._temperature,
        )
        response = completion.choices[0].message.content or ''
        if store_in_chat_queue and response:
            self.chat_queue.extend(messages)
        return response

    @classmethod
    def to_message_dict(
        cls,
        role: MessageRole,
        content: str,  # noqa: WPS110
    ) -> dict[str, str]:
        """Convert a message to a format recognized by the chat pipeline."""
        return {'role': str(role), 'content': content}
