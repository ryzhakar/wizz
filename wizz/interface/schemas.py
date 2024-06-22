from __future__ import annotations

import json
from io import StringIO
from string import Formatter
from typing import Self

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict

from wizz.interface import enums
from wizz.interface import types


class BaseYAMLConfig(BaseModel):
    """Base yaml-dumpable class."""
    model_config = ConfigDict(
        frozen=True,
    )

    def model_dump_yaml(self, *args, **kwargs) -> str:
        """Serialize as yaml."""
        serialized = self.model_dump_json(*args, **kwargs)
        return yaml.dump(json.loads(serialized))

    @classmethod
    def model_validate_yaml(cls, yaml_string: str) -> Self:
        """Deserialize from yaml."""
        return cls(**yaml.safe_load(StringIO(yaml_string)))


class Prompt(BaseYAMLConfig):
    """A message with a role and content."""
    role: enums.MessageRole
    content: str  # noqa: WPS110

    def to_tuple(self) -> types.MessageTuple:
        """Convert to a message tuple."""
        return self.role, self.content

    def get_formatted_copy(self, **kwargs) -> Self:
        """Return an instance with formatted content."""
        formattable_fields = {
            fieldname
            for _, fieldname, _, _ in Formatter().parse(self.content)
            if fieldname
        }
        if not formattable_fields:
            return self.model_copy()
        relevant_kwargs = {
            kwkey: kwval
            for kwkey, kwval in kwargs.items()
            if kwkey in formattable_fields
        }
        return self.model_copy(
            update={'content': self.content.format(**relevant_kwargs)},
        )

    @classmethod
    def from_tuple(cls, message_tuple: types.MessageTuple) -> Self:
        """Create from a message tuple."""
        role, content = message_tuple  # noqa: WPS110
        return cls(role=role, content=content)


class PromptChain(BaseYAMLConfig):
    """A chain of messages."""
    messages: list[Prompt]

    @classmethod
    def from_tuple_chain(
        cls,
        message_tuples: list[types.MessageTuple],
    ) -> Self:
        """Create from a chain of message tuples."""
        return cls(
            messages=[
                Prompt.from_tuple(message_tuple)
                for message_tuple in message_tuples
            ],
        )

    def to_tuple_chain(self) -> list[types.MessageTuple]:
        """Convert to a chain of message tuples."""
        return [message.to_tuple() for message in self.messages]

    def get_formatted_copy(self, **kwargs) -> Self:
        """Return an instance with formatted content."""
        new_messages = [
            message.get_formatted_copy(**kwargs) for message in self.messages
        ]
        return self.model_copy(
            update={'messages': new_messages},
        )
