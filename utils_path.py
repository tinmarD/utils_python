import os
import re

def createuniquedir(dirpath):
    """ Create a unique directory from a path. If .../.../dir_name already exists, .../.../dir_name_2 is returned.

    Parameters
    ----------
    dirpath : str
        Directory path

    Returns
    -------
    unique_dir_path : str
        Unique directory path

    """
    if os.path.exists(dirpath):
        dirpath = dirpath + '_2'
    inc = 3
    while os.path.exists(dirpath):
        dirpath = re.sub('_\d+$', '_{}'.format(inc), dirpath)
        inc += 1
    os.mkdir(dirpath)
    return dirpath
