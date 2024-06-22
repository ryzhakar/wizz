from collections.abc import Generator
from logging import getLogger

import tiktoken

from wizz.extraction.constants import CHUNK_OVERLAP
from wizz.extraction.constants import CHUNK_SIZE


logger = getLogger('wizz')


class TextBatcher:
    """Normalize string iterable to token-chunked batches with overlap."""

    def __init__(
        self,
        text: str,
        chunk_size_in_tokens: int = CHUNK_SIZE,
        chunk_overlap_in_tokens: int = CHUNK_OVERLAP,
    ) -> None:
        """Initializes the batcher with an iterable of strings."""
        logger.info('Initializing TextBatcher.')
        self.start_character_index = 0
        self.separator = ''
        self.buffer = text
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')
        self.chunk_size = chunk_size_in_tokens
        self.chunk_overlap = chunk_overlap_in_tokens

    def __iter__(self) -> Generator[tuple[int, str], None, None]:
        """Yield pairs of batch start indices and token-chunked text."""
        while True:
            tokens = self.tokenizer.encode(self.buffer)
            has_enough_tokens = len(tokens) >= self.chunk_size
            if not has_enough_tokens:
                break
            current_chunk_tokens, new_buffer = self._split_buffer(tokens)
            self.buffer = new_buffer
            batch = self.tokenizer.decode(current_chunk_tokens)
            yield self.start_character_index, batch
            self.start_character_index += len(batch)

        if self.buffer:
            yield self.start_character_index, self.buffer

    def _split_buffer(self, tokens: list) -> tuple[list, str]:
        """Splits the buffer into a chunk and a remainder."""
        split_point = self.chunk_size - self.chunk_overlap
        current_chunk_tokens = tokens[:self.chunk_size]
        remaining_tokens = tokens[split_point:]
        buffer = self.tokenizer.decode(remaining_tokens)
        return current_chunk_tokens, buffer
