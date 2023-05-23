from database import database
import numpy as np
import dask.array as da
import dask
import matplotlib.pyplot as plt

class visualization(database):
    def __init__(self, src):
        '''
        This class can plot any quantity from the database.
        '''
        database.__init__(self, src)
    
    # Somes ideas to (probably) implement
    '''
    # visu = cls(visualization), data = cls(database), calc = cls(calculations)
    visu.plot_mesh() # if time allows
    visu.rotate()    # if time allows
    # For the followings, possible to plot objects external to the db
    visu.plot_field() (where ? plan, 3D, coupe; when ?)   # if time allows
    visu.plot_statistics() (what ? averages, moments; where ? when ?)
    visu.plot_probabilities()  (what ? pdfs, joint_pdfs, correlations)
    '''