import unittest
import os


from utils.filesystem import read_text_file


class FilesystemTestCase(unittest.TestCase):
    def test_given_text_file_when_reading_then_its_content_is_read_as_string(self):
        test_file = 'test.txt'
        test_file_contents = 'contents'
        with open(test_file, 'w') as f:
            f.write(test_file_contents)

        actual_file_contents = read_text_file(test_file)

        os.remove(test_file)
        self.assertEqual(test_file_contents, actual_file_contents)
