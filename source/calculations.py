from database import Database
from stats import Stats
import os
from os import listdir
from os.path import isfile,isdir,join
import numpy as np
import dask
import dask.array as da
import pickle

class Calculations(Stats):
    def __init__(self, src, tree_dir):
        '''
        This class schedules calculations regarding a source database.
        The final results are gathered in an external one.
        src: original SFEMaNS db connector path
        tree_dir: calculations db connector path
        '''
        Calculations.__init__(self, src)
    