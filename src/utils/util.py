"""
Utilities
"""

import os
from typing import Callable

VALID_LABELS = {'PLAX': 0,
                'PSAX-ves': 1,
                'PSAX-base': 2,
                'PSAX-mid': 3,
                'PSAX-apical': 4,
                'Apical-2ch': 5,
                'Apical-3ch': 6,
                'Apical-5ch': 7,
                'Apical-4ch': 8,
                'Suprasternal': 9,
                'Subcostal': 10
                }
REQUIRES_AUG = [3, 4, 10, 6, 7] # PSAX-apical, PSAX-mid, suprasternal, apical-3ch, apical 5-ch


def calculate_accuracy(true_labels, predicted_labels) -> float:
    """
    Calculate the accuracy of a set of prediction labels

    Args:
        true_labels (Iterable[str]): The actual labels
        predicted_labels (Iterable[str]): The predicted labels
    
    Returns:
        The accuracy as a percentage in range [0.0, 1.0]
    """
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy

def assert_exists(path:str,
                  err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that there is a file or directory at the path
    
    Args:
        path (str): The path to check
        err_type (str -> Exception optional default: OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not exist
    """
    if not os.path.exists(os.path.abspath(path)):
        raise err_type(f'File/Folder "{os.path.abspath(path)}" does not exists')

def assert_is_dir(path:str,
                  err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that the path points to a directory
    
    Args:
        path (str): The path to check
        err_type (str -> Exception optional default: OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not point to a directory
    """
    if not os.path.exists(path):
        raise err_type(f'File/Folder "{path}" does not exists')

def assert_is_file(path:str,
                   readable:bool=False,
                   writable:bool=False,
                   err_type:Callable[[str], Exception]=OSError) -> None:
    """
    Verify that the path points to a file
    
    Args:
        path (str): The path to check
        readable (bool optional default: False): If true the file
            must additionally have read permission
        writable (bool optional default: False): If true the file
            must additionally have write permission
        err_type (str -> Exception optional default:OSError):
            A function that will return an Exception given a 
            error message
    Raises:
        err_type if the path does not point to a file or the file
        it points to does not have the correct permissions
    """
    if not os.path.isfile(path):
        raise err_type(f'"{path}" is not a file')
    if readable and not os.access(path, os.R_OK):
        raise err_type(f'"{path}" does not have read permission')
    if writable and not os.access(path, os.W_OK):
        raise err_type(f'"{path}" does not have write permission')
    
def log(*args, **kwargs):
    # if params.verbose:
    print(*args, **kwargs)
