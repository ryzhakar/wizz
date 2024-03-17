import re
from collections.abc import Generator
from collections.abc import Iterable
from logging import getLogger

import tiktoken

from wizz.extraction.constants import CHUNK_OVERLAP
from wizz.extraction.constants import CHUNK_SIZE


logger = getLogger('wizz')


class TextBatcher:
    """Normalize string iterable to token-chunked batches with overlap."""

    def __init__(
        self,
        texts: Iterable[str],
        chunk_size_in_tokens: int = CHUNK_SIZE,
        chunk_overlap_in_tokens: int = CHUNK_OVERLAP,
    ) -> None:
        """Initializes the batcher with an iterable of strings."""
        is_single_string = isinstance(texts, str)
        logger.info(
            'Initializing TextBatcher. Stream is a single text: %s',
            is_single_string,
        )
        self.separator = '' if is_single_string else ' '
        self.texts = texts
        self.buffer = ''
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')
        self.chunk_size = chunk_size_in_tokens
        self.chunk_overlap = chunk_overlap_in_tokens

    def __iter__(self) -> Generator[str, None, None]:
        """Yields token-chunked batches of text."""
        yield from map(self._postclean_text, self._iter())

    def _iter(self) -> Generator[str, None, None]:
        """Yields token-chunked batches of text."""
        for text in self.texts:
            text = self._preclean_text(text)
            self.buffer = (
                f'{self.buffer}{self.separator}{text}'
                if self.buffer
                else text
            )

            while True:
                tokens = self.tokenizer.encode(self.buffer)
                has_enough_tokens = len(tokens) >= self.chunk_size
                if not has_enough_tokens:
                    break
                current_chunk_tokens, new_buffer = self._split_buffer(tokens)
                self.buffer = new_buffer
                yield self.tokenizer.decode(current_chunk_tokens)

        if self.buffer:
            yield self.buffer

    def _split_buffer(self, tokens: list) -> tuple[list, str]:
        """Splits the buffer into a chunk and a remainder."""
        split_point = self.chunk_size - self.chunk_overlap
        current_chunk_tokens = tokens[:self.chunk_size]
        remaining_tokens = tokens[split_point:]
        buffer = self.tokenizer.decode(remaining_tokens)
        return current_chunk_tokens, buffer

    def _preclean_text(self, text: str) -> str:
        """Remove excess whitespace characters and prevent double spaces."""
        # First pass: Replace all sequences of whitespace
        # with a single space
        text = re.sub(r'\s+', ' ', text)
        # Second pass: Replace any double spaces
        # that might have resulted from the first pass
        text = re.sub(' {2,}', ' ', text)
        return text

    def _postclean_text(self, text: str) -> str:
        """Remove trailing and leading whitespace or special characters."""
        text = text.strip()
        special_characters = r'!@#$%^&*()_+\-=\[\]{};:"\\|,.<>\/?'  # noqa: P103
        # Remove leading special characters
        text = re.sub(f'^[{special_characters}]+', '', text)
        # Remove trailing special characters
        text = re.sub(f'[{special_characters}]+$', '', text)
        return text
