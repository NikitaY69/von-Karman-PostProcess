from database import Database
import numpy as np
import cupy as cp

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
        self.w = cp.array(self.db['v_weight'])
    '''
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fields input in all the class methods MUST already be converted to cupy arrays.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''

    def norm(self, field, slice=None):
        if slice is not None:
            field = self.advanced_slice(field, slice)
        return cp.linalg.norm(field, ord=2, axis=0, keepdims=True)
    
    def mean(self, field, penal=1, slice=None, type='spatial'):
        '''
        Computing of a mean quantity along a certain axis depending on the type of averaging.
        field must have the same structure as typical objects of the db (ie self.dims).
        If field was built on a peculiar slicing, the input must still respect self.dims
        NOTE: instead of slicing with A[:, 0, :, :], go with A[:, [0], :, :]
        '''
        self.mean_type_check(type)

        s = 1           # T axis
        w = None    
        if type != 'temporal':
            s = (2,3)   # P and N axis
            w = self.duplicate(self.w, field)
            if slice is not None:
                w = self.advanced_slice(w, slice)
            w *= penal
        if slice is not None:
            field = self.advanced_slice(field, slice)
        avg = cp.average(field, axis=s, weights=w)
        if type != 'both':
            return avg
        else:
            # if its 'both', you need to temporally average the spatial average
            return cp.average(avg, axis=None, weights=None)
    
    def moment(self, field, penal=1, slice=None, n=2, type='spatial', raw=True):
        # same principle as in mean function
        self.mean_type_check(type)

        if raw:
            avg = 0
        else:
            avg = self.mean(field, penal, slice, type)
            avg = self.duplicate(avg, field)
        to_compute = (field - avg)**n

        return self.mean(to_compute, penal, slice, type)

    '''
    For the pdfs, fields must be scalar ones.
    '''
    def pdf(self, field, penal=1, slice=None, bins=1000, range=None, save=None):
        '''
        Computes pdf(field).
        '''
        # general objects
        w = self.duplicate(self.w, field)
        w *= penal
        fields = [field, w]

        # slicing
        if slice is not None:
            fields = self.repeat_slice(fields, slice)

        # range
        if range is None:
            range = self.get_range(fields[0])
        
        field, w = fields
        H, edges = cp.histogram(field, bins=bins, range=range, weights=w, density=True)
        
        # saving
        if save is not None:
            H.tofile(save)

        return H, edges
    
    def joint_pdf(self, field1, field2, penal=1, slice=None, bins=1000, ranges=[None, None], \
                  log=True, save=None):
        '''
        Computes joint(field1, field2).
        field1 and field2 must have the same shape.
        NOTE: for limiting error propagation, if saving, it won't be log10.
        '''
        w = self.duplicate(self.w, field1)
        w *= penal
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
            fields[i] = cp.ravel(field)

        field1, field2, w = fields
        H, xedges, yedges = cp.histogram2d(field1, field2, bins=bins, range=ranges, \
                            weights=w**2, density=True)
        H = H.T

        # saving
        if save is not None:
            H.tofile(save)

        if log:
            H = cp.log10(H)

        return H, [xedges, yedges]

    def extract_pdf(self, joint, which, ranges):
        '''
        Extracts the marginals from a joint_pdf between (f1 and f2)
        which tells which marginal to extract:
        0 returns f1 while 1 return f2
        '''
        if which not in [0,1]:
            raise ValueError('which = 0 or 1. Assuming joint(f1, f2):\n\
                              which = 0 returns marg(f1)\n\
                              which = 1 returns marg(f2)')

        rev_range = {0:1, 1:0}
        f_range = self.compute_edges(joint.shape[0], [ranges[rev_range[which]]])[0]
        df = f_range[1]-f_range[0] # assuming linspaces!!!!

        return cp.sum(joint*df, axis=which)

    def conditional_avg(self, joint, which, penal=1, slice=None, bins=1000, ranges=[None,None], \
                        load=True):
        '''
        Computes E(A|B) where B is the field which lies on axis 'which' -
        joint being the joint_pdf of A and B.
        '''
        rev_range = {0:1, 1:0}
        if which not in [0,1]:
            raise ValueError('which = 0 or 1. Assuming joint(f1, f2):\n\
                              which = 0 returns E(f2|f1)\n\
                              which = 1 returns E(f1|f2)')
                              
        # gathering the pdfs
        if load:
            # you need to provide the ranges for the function to compute the pdfs
            if ranges[0] is None or ranges[1] is None:
                raise ValueError('When loading, you need to provide the ranges for each field.\
                                  Cannot compute the range based on an histogram only.')
            # extracting the marginal pdf of conditioned field
            pdf2 = self.extract_pdf(joint, which, ranges)

        else:
            # if not loading, joint = [field1, field2] 
            # with field1 (resp. field 2) lying on x-axis (resp. y-axis)
            field1, field2 = joint
            joint, [edges1, edges2] = self.joint_pdf(field1, field2, penal, slice, bins, ranges, False)
            # this in case ranges is initially [None, None]
            ranges = [[edges1[0], edges1[-1]], [edges2[0], edges2[1]]]
            pdf2 = self.extract_pdf(joint, which, ranges)

        # generate on whole meshgrid
        pdf2_f = cp.expand_dims(pdf2, axis=0)
        pdf2_f = cp.repeat(pdf2_f, bins, axis=0)

        # computing conditional probability
        if which == 1:
            pdf2_f = pdf2_f.T
        one_given_two = joint/pdf2_f
        one_given_two = cp.nan_to_num(one_given_two) # converting nans to 0s

        # expectation
        edges1 = self.compute_edges(bins, [ranges[rev_range[which]]])[0]
        d1 = edges1[1]-edges1[0] # delta field1 assuming linspaces!!!!
        ech1 = cp.array([(edges1[i]+edges1[i+1])/2 for i in range(bins)])
        # ech1 gathers the mid point of each bin from field1

        return cp.average(one_given_two, axis=which, weights=ech1*d1)*cp.sum(ech1*d1)

    def marginal_joint_pdf(self, joint, bins=1000, ranges=[None,None], log=False):
        '''
        Computes pdfA*pdfB on the whole meshgrid defining joint(A,B).
        It is only possible to compute the marginal_joint_pdf based on a loaded joint_pdf.
        (because the marginal_joint_pdf has no real physical purpose but to serve the correlations)
        '''
        # gathering the marginal pdfs
        # you need to provide the ranges for the function to compute the pdfs
        if ranges[0] is None or ranges[1] is None:
            raise ValueError('When loading, you need to provide the ranges for each field.\
                                Cannot compute the range based on an histogram only.')
        H1 = self.extract_pdf(joint, 0, ranges)
        H2 = self.extract_pdf(joint, 1, ranges)

        # Repeating on the whole meshgrid
        H1_f = cp.expand_dims(H1, axis=1)
        H2_f = cp.expand_dims(H2, axis=0)
        H1_f = cp.repeat(H1_f, bins, axis=1)
        H2_f = cp.repeat(H2_f, bins, axis=0)

        H = (H1_f*H2_f).T
        if log:
            H = cp.log10(H)

        return H

    def correlations(self, joint, penal=1, slice=None, bins=1000, ranges=[None,None], \
                     log=True, load=True):
        '''
        Computes joint(A,B)/pdfA*pdfB.
        '''
        if load:
            marginal = self.marginal_joint_pdf(joint, bins, ranges)
        else:
            # if not loading, joint = [field1, field2] 
            # with field1 (resp. field 2) lying on x-axis (resp. y-axis)
            field1, field2 = joint
            joint, [edges1, edges2] = self.joint_pdf(field1, field2, penal, slice, bins, ranges, False)
            # this in case ranges is initially [None, None]
            ranges = [[edges1[0], edges1[-1]], [edges2[0], edges2[1]]]
            marginal = self.marginal_joint_pdf(joint, bins, ranges)
            
        C = joint/marginal

        if log:
            C = cp.log10(C)
        
        return C

    def duplicate(self, field, target, rechunk=False):
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
        field2 = cp.expand_dims(field, axis=ax)

        # duplicating
        for i in ax:
            field2 = cp.repeat(field2, f2[i], axis=i)

        return field2

    '''
    The following 2 methods are class instances because of their close link
    to the database structure (1, T, P, N).
    '''
    def advanced_slice(self, field, condition):
        '''
        This function should only be used for advanced conditional slices on the mesh. 
        (not base index ones)
        '''
        if type(condition).__name__ != 'ndarray':
            raise NotImplementedError('You are probably giving a real slice.\n\
                                       This function only operating on the mesh;\
                                       please provide a condition outputting an array (ex: field<=3)')
        return field[:, :, :, cp.where(condition)[0]]

    def repeat_slice(self, fields, condition):
        for i, field in enumerate(fields):
            fields[i] = self.advanced_slice(field, condition)
        return fields

    @staticmethod    
    def load_reshaped(pdf, n):
        '''
        Reshapes a flattened pdf coming from np.fromfile to a n*n 2d pdf.
        '''
        H = cp.empty(shape=(n,n))
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
    def get_range(A):
        Amin = cp.nanmin(A)
        Amax = cp.nanmax(A)
        return [Amin, Amax]

    def key_check(self, key):
        # checks if a key is present in the db
        if key not in self.db.keys():
            raise ValueError('This key does not belong to the database.\n\
                              Please check self.db.keys().')
    
    def mean_type_check(self, type):
        if type not in ['spatial', 'temporal', 'both']:
            raise NotImplementedError('This type of average is not implemented. Please\
                              select one in ["spatial", "temporal", "both"].')