from functools import cache

from wizz.agent.chat import Chat
from wizz.interface import enums
from wizz.interface import schemas


@cache
def query_construction_message_chain() -> schemas.PromptChain:
    """Return a message chain object loaded from yaml."""
    with open('prompts/rag_construct_query.yaml') as promptfile:
        return schemas.PromptChain.model_validate_yaml(promptfile.read())


@cache
def search_integration_chain() -> schemas.PromptChain:
    """Return a message chain object loaded from yaml."""
    with open('prompts/rag_integrate_search.yaml') as promptfile:
        return schemas.PromptChain.model_validate_yaml(promptfile.read())


class Retriever(Chat):
    """Retrieval-augmented chat agent."""

    def construct_query(
        self,
        user_message: str,
    ) -> str:
        """Construct the query based on user history and user message.

        Stores only the user message in the chat history.
        """
        message = (enums.MessageRole.user, user_message)
        prompt_messages = query_construction_message_chain().to_tuple_chain()
        self.set_append_messages(*prompt_messages)
        return self.ask(message)

    def request_answer_based_on(
        self,
        *search_results: str,
        query: str,
    ) -> str:
        """Provide information about the query and context."""
        separator = '\n----\n'
        serialized_results = separator.join(search_results)
        messages = search_integration_chain().get_formatted_copy(
            search_query=query,
            search_results=serialized_results,
        ).to_tuple_chain()
        self.set_append_messages()
        answer = self.ask(
            *messages,
            store_in_chat_queue=False,
        )
        self.chat_queue.append((enums.MessageRole.assistant, answer))
        return answer
