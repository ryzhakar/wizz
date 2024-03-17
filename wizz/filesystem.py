import hashlib
import os
from collections.abc import Iterable

FileStream = Iterable[tuple[str, str, str]]


def shorten_filename(filename: str) -> str:
    """Shorten a filename to a maximum length."""
    cropdown = 25
    if len(filename) < cropdown + 3:
        return filename
    cropped = filename[:cropdown]
    return f'{cropped}...'


def get_file_streamer(directory: str) -> tuple[int, FileStream]:
    """Return the number if files and the streamer of files."""
    check_actions = [
        lambda fnm: os.path.isfile(os.path.join(directory, fnm)),
        lambda fnm: not fnm.startswith('.'),
        lambda fnm: fnm.endswith('.txt'),
    ]
    eligible_files = [
        filename
        for filename in os.listdir(directory)
        if all(check(filename) for check in check_actions)
    ]
    file_count = len(eligible_files)
    stream = stream_files_from(*eligible_files, directory=directory)
    return file_count, stream


def stream_files_from(*filenames: str, directory: str) -> FileStream:
    """Yield the name, content and the hash of each file in a directory."""
    for filename in filenames:
        with open(os.path.join(directory, filename)) as textfile:
            filecontent = textfile.read()
            yield filename, filecontent, hash_content(filecontent)


def hash_content(string_content: str) -> str:
    """Hash the content of a file."""
    return hashlib.sha1(
        string_content.encode('utf-8'),
        usedforsecurity=False,
    ).hexdigest()
