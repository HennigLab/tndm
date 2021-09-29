import os

from .logging import logger
from .empty_folder import empty_folder


def upsert_empty_folder(folder: dir):
    logger.debug('Creating %s folder' % (folder))
    try:
        os.mkdir(folder)
        logger.debug('Folder created')
    except FileExistsError:
        empty_folder(os.path.join(folder))
        logger.debug('Folder already exists')
