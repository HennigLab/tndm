import os
from .empty_folder import empty_folder


def remove_folder(folder: str):
    empty_folder(folder)
    os.rmdir(folder)
