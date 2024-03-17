import hashlib
import os
from collections.abc import Iterable


def stream_files_from(directory: str) -> Iterable[tuple[str, str, str]]:
    """Yield the name, content and the hash of each file in a directory."""
    check_actions = [
        lambda fnm: os.path.isfile(os.path.join(directory, fnm)),
        lambda fnm: not fnm.startswith('.'),
        lambda fnm: fnm.endswith('.txt'),
    ]
    eligible_files = (
        filename
        for filename in os.listdir(directory)
        if all(check(filename) for check in check_actions)
    )
    for filename in eligible_files:
        with open(os.path.join(directory, filename)) as textfile:
            filecontent = textfile.read()
            yield filename, filecontent, hash_content(filecontent)


def hash_content(string_content: str) -> str:
    """Hash the content of a file."""
    return hashlib.sha1(
        string_content.encode('utf-8'),
        usedforsecurity=False,
    ).hexdigest()
