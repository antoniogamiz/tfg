import glob
import zipfile
import requests
import os
from pathlib import Path

DATASET_DIR_PATH = '../datasets'
PATH_TO_ZIP_FILE = os.path.join(DATASET_DIR_PATH, 'dataset.zip')

DATASET_NEW_FILE_NAME = 'dataset.json'
DATASET_NEW_PATH = os.path.join(DATASET_DIR_PATH, DATASET_NEW_FILE_NAME)

DEFAULT_DATASET_URL = 'https://github.com/rhgarcia/tropescraper/raw/master/datasets/tvtropes_20200302.json.zip'


def main():
    clean()
    url = os.getenv('DATASET_URL', DEFAULT_DATASET_URL)
    _download_file(url, PATH_TO_ZIP_FILE)
    _extract_zip_file(PATH_TO_ZIP_FILE, DATASET_DIR_PATH)
    os.remove(PATH_TO_ZIP_FILE)
    original_dataset_name = _get_dataset_json_file()
    os.rename(original_dataset_name, DATASET_NEW_PATH)


def _download_file(url, save_to):
    response = requests.get(url)
    dataset_directory = Path(save_to).parts[-2]
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
    with open(save_to, 'wb')as f:
        f.write(response.content)


def _extract_zip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def _get_dataset_json_file():
    return glob.glob(f'{DATASET_DIR_PATH}/*.json')[0]


def clean():
    files = glob.glob(f'{DATASET_DIR_PATH}/*.*')
    for f in files:
        os.remove(f)
