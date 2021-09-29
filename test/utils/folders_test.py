import os
import pytest

from tndm.utils import upsert_empty_folder, empty_folder, remove_folder


@pytest.fixture(scope='function')
def tmp_folder():
    dir_name = os.path.join('.', 'test', 'utils', 'tmp')
    upsert_empty_folder(dir_name)
    upsert_empty_folder(os.path.join(dir_name, 'inner_folder'))
    return dir_name


@pytest.fixture(scope='function', autouse=True)
def cleanup(request, tmp_folder):
    def remove_test_dir():
        remove_folder(tmp_folder)
    request.addfinalizer(remove_test_dir)


@pytest.mark.unit
@pytest.mark.smoke
def test_empty_folder(tmp_folder):
    empty_folder(tmp_folder)


@pytest.mark.unit
@pytest.mark.smoke
def test_upsert_folder(tmp_folder):
    upsert_empty_folder(tmp_folder)
