def read_text_file(path: str) -> str:
    with open(path) as f:
        file_contents = f.read()

    if isinstance(file_contents, bytes):
        return file_contents.decode('utf-8')

    return file_contents
