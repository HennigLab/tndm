import os
import shutil
from latentneural.utils import logger


test_notebooks_folder = os.path.join('test', 'notebooks', 'deliverables')
original_notebooks_folder = os.path.join('notebooks', 'deliverables')

def empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """

    logger.info('Creating %s folder' % (test_notebooks_folder))
    try:
        os.mkdir(test_notebooks_folder)
        logger.info('Folder created')
    except FileExistsError:
        empty_folder(os.path.join(test_notebooks_folder))
        logger.info('Folder already exists')

    logger.info('Finding deliverables notebooks')
    notebooks = [f for f in os.listdir(original_notebooks_folder) if os.path.isfile(os.path.join(original_notebooks_folder, f)) and f[-6:] == '.ipynb']

    for notebook in notebooks:
        os.system('jupyter nbconvert --to python --output "%s" "%s"' % (
            os.path.join('..', '..', test_notebooks_folder, notebook).replace('.ipynb', '.py'),
            os.path.join(original_notebooks_folder, notebook),
        ))
        
def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    empty_folder(test_notebooks_folder)
    os.rmdir(test_notebooks_folder)