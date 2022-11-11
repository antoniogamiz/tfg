import unittest
import os

from scripts.get_data_set import main, DATASET_NEW_PATH, PATH_TO_ZIP_FILE


class GetDataSetTestCase(unittest.TestCase):
    def test_get_data_set(self):
        main()

        self.assertTrue(os.path.exists(DATASET_NEW_PATH), 'JSON dataset is created')
        self.assertFalse(os.path.exists(PATH_TO_ZIP_FILE), 'Zip file was deleted')
