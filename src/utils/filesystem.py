import os


def delete_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def create_directory_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def read_text_file(path: str) -> str:
    with open(path) as f:
        file_contents = f.read()

    if isinstance(file_contents, bytes):
        return file_contents.decode('utf-8')

    return file_contents
