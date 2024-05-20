import hydra
import os

from pathlib import Path
from typing import Union

def convert_to_posix(path: Union[str, Path], 
                     project_dir: Union[str, Path]) -> Path:
    """
    Converts a given path to a POSIX path. If the path is relative, 
    it is converted to an absolute path using the project directory.
    Args:
        path (_type_): _description_
        project_dir (_type_): _description_
    """
    
    if path.startswith('/') or path.startswith('\\') or path.startswith('~'):
        return Path(path)
    
    else:
        return Path(project_dir) / path
    