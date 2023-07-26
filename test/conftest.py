import pytest
from import_utils import *
import tabensemb
import shutil


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        path = tabensemb.setting["default_output_path"]
        if os.path.exists(path):
            shutil.rmtree(path)

    request.addfinalizer(remove_test_dir)
