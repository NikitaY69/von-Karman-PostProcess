from database import Database
import numpy as np
import dask.array as da
import dask

class Stats(Database):
    def __init__(self, src):
        '''
        This class incorporates useful computations related to the statistics of 
        1-component fields.
        Typical dimensions for 1-component fields are (1, T, P, N_P)
        (Reminder: 3-components fields are concatenations of the base ones)
        '''
        Database.__init__(self, src)
        dims = list(self.db['u']['field'].shape) # u will always be present in any db
        dims[0] = int(dims[0]/3)
        self.dims = tuple(dims) # 0, 1, 2, 3 --> component, times, planes, mesh coordinates indices

        self.base_chunk = self.db['u']['field'].chunksize

    def norm(self, field, slice=None):
        mod = self.set_module(field)
        if slice is not None:
            field = self.advanced_slice(field, slice)
        return mod.linalg.norm(field, ord=2, axis=0, keepdims=True)
    
    def mean(self, field, slice=None, type='spatial'):
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
            if slice is not None:
                w = self.advanced_slice(w, slice)

        if slice is not None:
            field = self.advanced_slice(field, slice)
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
    '''
    def pdf(self, field, slice=None, bins=1000, range=None, save=None):

        # general objects
        mod = self.set_module(field)
        w = self.duplicate(self.db['v_weight'], field)
        fields = [field, w]

        # slicing
        if slice is not None:
            fields = self.repeat_slice(fields, slice)
        
        # range
        if range is None:
            range = self.get_range(field)

        if mod is da:
            fields = self.prepare(fields)

        field, w = fields
        H, edges = mod.histogram(field, bins=bins, range=range, weights=w, density=True)
        # H = H.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...
        
        # saving
        if save is not None:
            H.tofile(save)

        return H, edges

    def joint_pdf(self, field1, field2, slice=None, bins=1000, ranges=[None, None], \
                  log=True, save=None):
        '''
        field1 and field2 must have the same shape.
        Note: for limiting error propagation, if saving, it won't be log10.
        '''
        mod = self.set_module(field1)
        w = self.duplicate(self.db['v_weight'], field1)
        fields = [field1, field2, w]

        # slicing
        if slice is not None:
            fields = self.repeat_slice(fields, slice)

        # range
        for i, r in enumerate(ranges):
            if r is None: 
                ranges[i] = self.get_range(fields[i])

        # flattening the data    
        for i, field in enumerate(fields):
            fields[i] = mod.ravel(field)
        
        if mod is da:
             fields = self.prepare(fields)
        
        field1, field2, w = fields
        H, xedges, yedges = mod.histogram2d(field1, field2, bins=bins, range=ranges, \
                            weights=w**2, density=True)
        H = H.T
        # H = H.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        # saving
        if save is not None:
            H.tofile(save)

        if log:
            H = mod.log10(H)

        return H, [xedges, yedges]

    def marginal_joint_pdf(self, field1, field2, slice=None, bins=1000, ranges=[None,None],\
                           log=False, load=True, all=False):

        # gathering the marginal pdfs
        if load:
            # in this case, field1 and field2 are already the marginal pdfs NOT the fields
            H1, H2 = field1, field2
            # you need to provide the ranges for the function to compute the edges
            if ranges[0] is None or ranges[1] is None:
                raise ValueError('When loading, you need to provide the ranges for each field.\
                                  Cannot compute the range based on an histogram only.')
            edges1, edges2 = self.compute_edges(bins, ranges)

        else:
            H1, edges1 = self.pdf(field1, slice, bins, ranges[0])
            H2, edges2 = self.pdf(field2, slice, bins, ranges[1])

        mod = self.set_module(field1)

        # Repeating on the whole meshgrid
        H1_f = mod.expand_dims(H1, axis=1)
        H2_f = mod.expand_dims(H2, axis=0)
        H1_f = mod.repeat(H1_f, len(H2), axis=1)
        H2_f = mod.repeat(H2_f, len(H1), axis=0)
        if mod is da:
            H1_f = H1_f.rechunk(bins//2.5)
            H2_f = H2_f.rechunk(bins//2.5)
        # there is probably a more clever way to rechunk ...

        H = (H1_f*H2_f).T
        if log:
            H = mod.log10(H)

        if all:
            return H, [edges1, edges2], [H1, H2]
        else:
            return H, [edges1, edges2]

    def correlations(self, field1, field2, slice=None, bins=1000, ranges=[None,None], log=True,\
                     load=True, all=False):
        '''
        If loading, field1 is the joint_pdf and field2 = [field1, field2],
        where field1 and field2 are the real fields of interest.
        This is really bad.
        '''
        args = [field1, field2, slice, bins, ranges, False, load, all]
        if load:
            args[0], args[1] = field2
            joint = self.load_reshaped(field1, bins)
            marginal = self.marginal_joint_pdf(*args)
        else:
            joint = self.joint_pdf(*args[:-2])[0]
            marginal = self.marginal_joint_pdf(*args)

        C = joint/marginal[0]

        mod = self.set_module(field1)
        if log:
            C = mod.log10(C)
        
        if not all:
            return C, marginal[1] 
            # Corr - [edges]
        else:
            return C, marginal[1], joint, marginal[2]
            # Corr - [edges] - joint_pdf - [marginal_pdfs]

    def duplicate(self, field, target, rechunk=False):
        # this function is useful to reshape a field properly to
        # a target array (for the purpose of dealing with same dimensions)
        
        mod = self.set_module(target)
        # some objects in the db are not naturally dask arrays
        if type(field).__name__ == 'ndarray' and mod is da:
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

    '''
    The following 2 methods are class instances because of their close link
    to the database structure (1, T, P, N).
    '''
    def advanced_slice(self, field, condition):
        '''
        This function should only be used for advanced conditional slices on the mesh. 
        (not base index ones)
        np.where is mandatory for dask, it is not for traditional ndarrays.
        '''
        if type(condition).__name__ != 'ndarray':
            raise NotImplementedError('You are probably giving a real slice.\n\
                                       This function only operating on the mesh;\
                                       please provide a condition outputting an array (ex: field<=3)')
        return field[:, :, :, np.where(condition)[0]]

    def repeat_slice(self, fields, condition):
        for i, field in enumerate(fields):
            fields[i] = self.advanced_slice(field, condition)

    @staticmethod    
    def load_reshaped(pdf, n):
        '''
        Reshapes a flattened pdf coming from np.fromfile to a n*n 2d pdf.
        '''
        H = np.empty(shape=(n,n))
        # rebuilding
        for l in range(n):
            H[l, :] = pdf[l*n:(l+1)*n]
        return H

    @staticmethod
    def compute_edges(bins, ranges):
        edges = []
        for r in ranges:
            edges.append(np.linspace(r[0], r[1], bins+1))
        return edges

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

    @staticmethod
    def get_range(A):
        Amin = A.min()
        Amax = A.max()
        if type(A).__name__ == 'Array':
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
        elif type(field).__name__ == 'ndarray':
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