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
    def __init__(self, src):
        '''
        This class schedules calculations regarding a source database.
        The final results are gathered in an external.
        '''

