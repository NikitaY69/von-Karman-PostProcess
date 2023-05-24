from database import database
import numpy as np
import dask.array as da
import dask

class calculations(database):
    def __init__(self, src):
        '''
        This class incorporates useful computations related to the statistics of 
        1-component fields.
        Typical dimensions for 1-component fields are (1, T, P, N_P)
        (Reminder: 3-components fields are concatenations of the base ones)
        All the calculations here only yield a dask task. You need to compute
        them afterwards. 
        The parallelization is automatically made in the dask graphs/chunks (I think).
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
        '''
        self.mean_type_check(type)
 
        s = 1           # T axis
        w = None    
        if type != 'temporal':
            s = (2,3)   # P and N axis
            w = self.duplicate(da.from_array(self.db['v_weight']), field).rechunk(
                               self.base_chunk)
        avg = da.average(field, axis=s, weights=w, keepdims=True)
        if type != 'both':
            return avg
        else:
            # if its 'both', you need to temporally average the spatial average
            return da.average(avg, axis=None, weights=None, keepdims=True)
    
    def moment(self, field, type, n, raw=True):
        # same principle as in mean function
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
    TO-DO:
    - Study the importance of the choice of bins
    - Rechunk more cleverly
    '''
    def pdf(self, field, bins=1000, range=None):
        # histogram range should be offseted case by case after visualization
        w = self.duplicate(da.from_array(self.db['v_weight']), field).rechunk(
                           self.base_chunk)
        if range is None:
            range = self.get_range(field)

        H, edges = da.histogram(field, bins=bins, range=range, weights=w, density=True)
        H = H.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        return H, edges

    def joint_pdf(self, field1, field2, bins=1000, ranges=[None, None], log=True):
        w = self.duplicate(da.from_array(self.db['v_weight']), field1).rechunk(
                           self.base_chunk)
        fields = [field1, field2]

        # flattening the data
        w_l = da.ravel(w)
        f1_l = da.ravel(field1)
        f2_l = da.ravel(field2)

        for i, r in enumerate(ranges):
            if r is None: 
                ranges[i] = self.get_range(fields[i])
            else:
                ranges[i] = ranges[i]
        
        H, xedges, yedges = da.histogram2d(f1_l, f2_l, bins=bins, range=ranges, \
                              weights=w_l**2, density=True)
        H = (H.T).rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        if not log:
            return H, [xedges, yedges]
        else:
            return da.log10(H), [xedges, yedges]
        
    def marginal_joint_pdf(self, field1, field2, bins=1000, ranges=[None,None], log=True):
        # computing the marginal pdfs
        H1, edges1 = self.pdf(field1, bins, ranges[0])
        H2, edges2 = self.pdf(field2, bins, ranges[1])

        # Repeating on the whole meshgrid
        H1_f = da.expand_dims(H1, axis=1)
        H2_f = da.expand_dims(H2, axis=0)
        H1_f = da.repeat(H1_f, len(H1), axis=1).rechunk(bins//2.5)
        H2_f = da.repeat(H2_f, len(H2), axis=0).rechunk(bins//2.5)

        H = (H1_f*H2_f).T
        if log:
            H = da.log10(H)

        return H, [edges1, edges2], [H1, H2]

    def correlations(self, field1, field2, bins=1000, ranges=[None,None], log=True,\
                     all=False):
        args = (field1, field2, bins, ranges, False)
        joint = self.joint_pdf(*args)
        marginal = self.marginal_joint_pdf(*args)
        C = joint[0]/marginal[0]
        if log:
            C = da.log10(C)
        
        if not all:
            return C, joint[1] 
            # Corr - [edges]
        else:
            return C, joint[1], joint[0], [marginal[0], marginal[2]]
            # Corr - [edges] - joint_pdf - [marginal_joint_pdf, marginal_pdfs]

    @staticmethod
    def duplicate(field, target, rechunk=False):
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
        if rechunk:
            # new_chunk = tuple([1 if d not in ax else self.base_chunk[d] for d in range(4)])
            new_chunk = field2.shape
            field2 = field2.rechunk(new_chunk)
        # duplicating
        for i in ax:
            field2 = da.repeat(field2, f2[i], axis=i)

        return field2
        

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