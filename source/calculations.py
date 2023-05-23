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
        '''
        database.__init__(self, src)
        dims = list(self.db['u']['field'].shape) # u will always be present in any db
        dims[0] = int(dims[0]/3)
        self.dims = tuple(dims) # 0, 1, 2, 3 --> component, times, planes, mesh coordinates indices

        self.base_chunk = self.db['u']['field'].chunksize

    def norm(self, key):
        # computes the norm of a field present in the db
        self.key_check(key)
        return da.linalg.norm(self.db[key]['field'], ord=2, axis=0, keepdims=True)
    
    def mean(self, field, type):
        '''
        Computing of a mean quantity along a certain axis depending on the type of averaging.
        field must have the same structure as typical objects of the db (ie self.dims).
        If field was built on a peculiar slicing, the input must still respect self.dims
        (tip: instead of slicing with A[:, 0, :, :], go with A[:, [0], :, :])

        Note for improvement:
        I need to rethink the database so that computed quantities can also be stored.
        For example, when you compute n-th non-raw moments, it will be stupid to compute
        the mean for each computation ...
        Field must be a 1-component dask array.
        '''
        self.mean_type_check(type)
 
        s = 1           # T axis
        w = None    
        if type != 'temporal':
            s = (2,3)   # P and N axis
            w = self.duplicate(da.from_array(self.db['v_weight']), field)
           
        avg = da.average(field, axis=s, weights=w)
        if type != 'both':
            return avg
        else:
            # if its 'both', you need to temporally average the spatial average
            return da.average(avg, axis=None, weights=None)
    
    def moment(self, field, type, n, raw=True):
        # same principe as in mean function
        self.mean_type_check(type)

        if raw:
            avg = 0
        else:
            avg = self.mean(field, type)
            avg = self.duplicate(avg, field)

        to_compute = (field - avg)**n

        return self.mean(to_compute, type)

    '''
    For the pdfs, fields must be scalar ones having SAME shape.
    '''
    def pdf(self, field, bins=1000, range=None):
        # histogram range should be offseted case by case after visualization
        w = self.duplicate(da.from_array(self.db['v_weight']), field).rechunk(
                           self.base_chunk)
        if range is None:
            range = self.get_range(field)

        return da.histogram(field, bins=bins, range=range, weights=w, density=True)

    def joint_pdf(self, field1, field2, bins=1000, range=None, log=True):
        w = self.duplicate(da.from_array(self.db['v_weight']), field1).rechunk(
                           self.base_chunk)
        w_l = da.ravel(w)
        f1_l = da.ravel(field1)
        f2_l = da.ravel(field2)

        if range is None:
            range1 = self.get_range(field1)
            range2 = self.get_range(field2)
            range = [range1, range2]
        
        H, xedges, yedges = da.histogram2d(f1_l, f2_l, bins=bins, range=range, \
                              weights=w_l**2, density=True)

        if not log:
            return H, xedges, yedges
        else:
            H = H.rechunk(bins//2.5)   # to probably improve (it fits for 1000)
            return da.log10(H), xedges, yedges
        
        
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

    @staticmethod
    def get_range(A):
        # A must be a dask array!
        Amin = A.min().compute()
        Amax = A.max().compute()
        return [Amin, Amax]

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