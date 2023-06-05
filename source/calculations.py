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
        '''
        database.__init__(self, src)
        dims = list(self.db['u']['field'].shape) # u will always be present in any db
        dims[0] = int(dims[0]/3)
        self.dims = tuple(dims) # 0, 1, 2, 3 --> component, times, planes, mesh coordinates indices

        self.base_chunk = self.db['u']['field'].chunksize

    def norm(self, field):
        mod = self.set_module(field)
        return mod.linalg.norm(field, ord=2, axis=0, keepdims=True)
    
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
        mod = self.set_module(field)

        s = 1           # T axis
        w = None    
        if type != 'temporal':
            s = (2,3)   # P and N axis
            w = self.duplicate(self.db['v_weight'], field)
        avg = mod.average(field, axis=s, weights=w)
        if type != 'both':
            return avg
        else:
            # if its 'both', you need to temporally average the spatial average
            return mod.average(avg, axis=None, weights=None)
    
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
    For the pdfs, fields must be scalar ones.
    TO-DO:
    - Rechunk more cleverly
    '''
    def pdf(self, field, bins=1000, range=None):
        # histogram range should be offseted case by case after visualization
        mod = self.set_module(field)
        
        w = self.duplicate(self.db['v_weight'], field)
        if range is None:
            range = self.get_range(field)

        if mod is da:
            fields = [field, w]
            field, w = self.prepare(fields)

        H, edges = mod.histogram(field, bins=bins, range=range, weights=w, density=True)
        # H = H.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        return H, edges

    def joint_pdf(self, field1, field2, bins=1000, ranges=[None, None], log=True):
        '''
        field1 and field2 must have the same shape.
        '''
        mod = self.set_module(field1)
        w = self.duplicate(self.db['v_weight'], field1)
        fields = [field1, field2]
        for i, r in enumerate(ranges):
            if r is None: 
                ranges[i] = self.get_range(fields[i])

        # flattening the data    
        w_l = mod.ravel(w)
        f1_l = mod.ravel(field1)
        f2_l = mod.ravel(field2)

        if mod is da:
             fields = [f1_l, f2_l, w_l]
             f1_l, f2_l, w_l = self.prepare(fields)

        H, xedges, yedges = mod.histogram2d(f1_l, f2_l, bins=bins, range=ranges, \
                            weights=w_l**2, density=True)
        H = H.T
        # H = H.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...
        if log:
            H = mod.log10(H)

        return H, [xedges, yedges]


    def marginal_joint_pdf(self, field1, field2, bins=1000, ranges=[None,None], log=True):
        # computing the marginal pdfs
        H1, edges1 = self.pdf(field1, bins, ranges[0])
        H2, edges2 = self.pdf(field2, bins, ranges[1])

        mod = self.set_module(field1)

        # Repeating on the whole meshgrid
        H1_f = mod.expand_dims(H1, axis=1)
        H2_f = mod.expand_dims(H2, axis=0)
        H1_f = mod.repeat(H1_f, len(H1), axis=1)
        H2_f = mod.repeat(H2_f, len(H2), axis=0)
        if mod is da:
            H1_f = H1_f.rechunk(bins//2.5)
            H2_f = H2_f.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        H = (H1_f*H2_f).T
        if log:
            H = mod.log10(H)

        return H, [edges1, edges2], [H1, H2]

    def correlations(self, field1, field2, bins=1000, ranges=[None,None], log=True,\
                     all=False):
        args = (field1, field2, bins, ranges, False)
        joint = self.joint_pdf(*args)
        marginal = self.marginal_joint_pdf(*args)
        C = joint[0]/marginal[0]

        mod = self.set_module(field1)
        if log:
            C = mod.log10(C)
        
        if not all:
            return C, joint[1] 
            # Corr - [edges]
        else:
            return C, joint[1], joint[0], [marginal[0], marginal[2]]
            # Corr - [edges] - joint_pdf - [marginal_joint_pdf, marginal_pdfs]

    @staticmethod
    def prepare(tasks, persist=True):
        '''
        Prepare a set of tasks into a future object.
        Each task must come from dask.

        Note: it might be possible to parallelize here with dask.compute(*tasks)
              but I don't know if it's a good idea + if it holds with persist method.
        '''
        if persist:
            for i, task in enumerate(tasks):
                tasks[i] = task.persist()
        else:
            for i, task in enumerate(tasks):
                tasks[i] = task.compute()

        return tasks

    def duplicate(self, field, target, rechunk=False):
        # this function is useful to reshape a field properly to
        # a target array (for the purpose of dealing with same dimensions)
        
        mod = self.set_module(target)
        # some objects in the db are not naturally dask arrays
        if type(field).__name__ is 'ndarray' and mod is da:
            field = da.from_array(field)

        # finding dimensions on which to copy initial array
        f1 = field.shape
        f2 = target.shape
        ax = []
        for i, d in enumerate(f2):
            if d not in f1:
                ax.append(i)
        ax = tuple(ax)
        field2 = mod.expand_dims(field, axis=ax)

        # duplicating
        for i in ax:
            field2 = mod.repeat(field2, f2[i], axis=i)
        if mod is da:
            if rechunk:
                new_chunk = field2.shape
            else:
                new_chunk = self.base_chunk
            field2 = field2.rechunk(new_chunk)

        return field2

    @staticmethod
    def get_range(A):
        Amin = A.min()
        Amax = A.max()
        if type(A).__name__ is 'Array':
            Amin = Amin.compute()
            Amax = Amax.compute()
        return [Amin, Amax]

    @staticmethod
    def set_module(field):
        '''
        This function checks the nature of an object and selects the module which
        fits best for calculations related to it.
        '''
        if type(field).__name__ not in ['ndarray', 'Array']:
            raise NotImplementedError("It is not possible to deal with this kind of field.\n\
                                       It must be either delayed through Dask or simply a numpy array.")
        elif type(field) is 'ndarray':
            return np
        else:
            return da

    def key_check(self, key):
        # checks if a key is present in the db
        if key not in self.db.keys():
            raise ValueError('This key does not belong to the database.\n\
                              You cannot compute the norm of a field external to the db\
                              (for example, a field you manually builded).\n\
                              Please check self.db.keys().')
    
    def mean_type_check(self, type):
        if type not in ['spatial', 'temporal', 'both']:
            raise NotImplementedError('This type of average is not implemented. Please\
                              select one in ["spatial", "temporal", "both"].')