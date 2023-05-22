from database import database
import numpy as np
import dask.array as da
import dask

class calculations(database):
    def __init__(self, src):
        '''
        This class incorporates useful computations related to the statistics of a database.
        Typical dimensions for 1-component fields are (1, T, P, N_P)
        Reminder: 3-components fields are concatenations of the base ones.
        All the calculations here only compute a dask task. You need to compute
        them afterwards.
        Note for improvement:
        All these init definitions can already be given in database..
        '''
        database.__init__(self, src)
        dims = list(self.db['u']['field'].shape) # u will always be present in any db
        dims[0] = int(dims[0]/3)
        self.dims = tuple(dims) # 0, 1, 2, 3 --> component, times, planes, mesh coordinates indices

        self.base_chunk = self.db['u']['field'].chunksize
        # self.global_weight = self.duplicate(self.db['v_weight'], self.dims).rechunk(self.base_chunk)

    def norm(self, key):
        # computes the norm of a field present in the db
        self.key_check(key)
        return da.linalg.norm(self.db[key]['field'], ord=2, axis=0, keepdims=True)
    
    def mean(self, field, type):
        '''
        Computing of a mean quantity along a certain axis depending on the type of averaging.
        Note for improvement:
        I need to rethink the database so that computed quantities can also be stored.
        For example, when you compute n-th moments, it will be stupid to compute
        the mean for each computation ...
        Field must be a 1-component dask array.
        '''
        self.mean_type_check()

        if type == 'spatial':
            s = (2,3) # planes and coordinates
            w = self.duplicate(self.db['v_weight'], field.shape)
        elif type == 'temporal':
            s = 1     # time axis
            w = None            
        avg = da.average(field, axis=s, weights=w)
        if type != 'both':
            return avg
        else:
            # if its 'both', you need to temporally average the spatial average
            return da.average(avg, axis=None, weights=None)
    
    def nth_moment(self, field, type, n):
        # same principe as in mean function
        self.mean_type_check()
        avg = self.mean(field, type)

        if type == 'spatial':
            s = (2,3) # planes and coordinates
            w = self.duplicate(self.db['v_weight'], field.shape)
            avg2 = self.duplicate(avg, field.shape, type)
        elif type == 'temporal':
            s = 1     # time axis
            w = None
            avg2 = self.duplicate(avg, field.shape, type)           

        moment = da.average((field-avg)**n, axis=s, weights=w)
        if type != 'both':
            return moment
        else:
            # if its 'both', you need to temporally average the spatial average
            return da.average(moment, axis=None, weights=None)
        
    '''
    For the pdfs, fields must be scalar ones having SAME shape.
    '''
    def pdf(self, field, bins=1000, range=None):
        # histogram range should be reajusted case by case after visualization
        # for now on, it is just [min, max]
        return da.histogram(field, bins=bins, range=range, weights=self.global_weight, density=True)
    
    def joint_pdf(self, field1, field2, bins=1000, range=None):
        return da.histogram2d(field1, field2, range=None, weights=self.global_weight**2, density=True)
    # vor visualization, it is better to scale logarithmically

    @staticmethod
    def duplicate(field, target):
        # this function is useful to reshape a field properly to
        # a target array (for the purpose of dealing with same dimensions)
        # finding dimensions on which to copy initial array
        f1 = field.shape
        f2 = target.shape
        ax = []
        for i, d in enumerate(f2):
            if d not in f1:
                ax.append(i)
        ax = tuple(ax)
        field2 = da.expand_dims(field, axis=ax)
        
        # duplicating
        for i in ax:
            field2 = da.repeat(field2, f2[i], axis=i)

        return field2
        
    # add function to automatically rechunk an object based on the axis the initial
    # object was duplicated

    def key_check(self, key):
        # checks if a key is present in the db
        if key not in self.db.keys():
            raise ValueError('This key does not belong to the database.\n\
                              You cannot compute the norm of a field external to the db\
                              (for example, a field you manually builded).\n\
                              Please check self.db.keys().')
        else:
            pass
    
    def mean_type_check(self, type):
        if type not in ['spatial', 'temporal', 'both']:
            raise NotImplementedError('This type of average is not implemented. Please\
                              select one in ["spatial", "temporal", "both"].')
        else:
            pass