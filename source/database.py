import os
from os import listdir
from os.path import isfile,isdir,join
import numpy as np
import dask
import dask.array as da
import pickle

class database():
    '''
    Please note that all the functions provided in this class were rawly written by 
    Rémi Bousquet. I have just reframed them in a more object-oriented manner.
    '''
    def __init__(self, src, **kwargs):
        '''
        Initiation of the database.
        '''
        try:
            # directly loading
            self.db = self.load_db(src)
        except:
            # if not, create a first db
            self.raw_path = src
            self.db_path = kwargs.get('target')

            self.M = kwargs.get('M')
            self.P = 2*self.M - 1 # phys

            # misc params for db creation
            sfx = kwargs.get('sfx')
            dict_name = kwargs.get('dic')
            space = kwargs.get('space', 'phys') # 'phys' or 'fourier'
            self.make(sfx, dict_name, space)

    def load_db(self, src):
        return pickle.load(open(src, 'rb'))

    def make(self, sfx, dict_name, space):
        '''
        Please note that all the data dirs must be subs of the raw_path tree structure.
        '''
        global_path = self.raw_path + sfx
        dict_path = self.db_path + dict_name
        sort_with = {"r" : 0 , "z" : 1, "w":2}
        sorting_method = sort_with["z"]
        # sorry about the following
        P = self.P
        if space == "fourier":
            P = self.M

        # add experiment to directory if it does not exist
        self.verif_exp(self.db_path, dict_name)        

        # extract the experiment files names
        field_files = np.sort([f for f in listdir(global_path) if isdir(join(global_path, f))])
        axes_files = np.sort([f for f in listdir(global_path) if not isdir(join(global_path, f))])
        
        # get spatio-temporal and weight axes
        v_axes, h_axes, t_axis, vN_per_S, hN_per_S = self.get_axes(global_path, self.raw_path,
                                                                   axes_files)
        v_args = np.argsort(np.vstack(v_axes[:2]))
        h_args = np.argsort(np.vstack(h_axes[:2]))
        
        # save spatio-temporal and weight axes
        self.save_axes(sorting_method, dict_path, v_axes, h_axes, v_args, h_args) 
        for field_name in field_files:
            # get_field files and info
            path_to_field = join(global_path, field_name)
            files, _, (D,S,T) = self.get_field_info(path_to_field, space) 
            N =  [self.get_file_length(join(path_to_field,files[T*i]))//P for i in range(S) ]  
            data_struct = (D,S,T,P,N)
            # get dask view of the field shape is (D,T,P,Ntot), domains have been concatenated
            field = self.mmap_phys_field(path_to_field,files,data_struct)
            
            #spatial sort
            if sum(N) == sum(vN_per_S):
                # mesh is velocity mesh
                field = (field.T[v_args[sorting_method]]).T
            elif sum(N) == sum(hN_per_S):
                # mesh is magnetic field mesh
                field = (field.T[h_args[sorting_method]]).T
            else:
                print(f"Mesh of field {field_name} unknown, it will not be spatially sorted.")
                redflag = input("Process anyway ? (if n the field will be ignored) (y/n) : ")
                if redflag != "y" or redflag != "yes":
                    continue


            # save the dask file view to dictionnary
            experiment = pickle.load(open(dict_path, "rb"))
            try : # try to add the data to existing field data (as a suite)
                # get existing field
                expe_field = experiment[field_name]["field"]
                expe_time  = experiment[field_name]["time_axis"]
                
                # security : try if the data already been addded
                doubled_frame = np.sum([i==j for i in expe_time for j in t_axis])
                if doubled_frame == t_axis.size: 
                    print("Data of field", field_name, "was already present. \
                          It will not be added again.")
                    continue 
                if doubled_frame >=1:
                    print("Data of field", field_name, "corrupted :", doubled_frame, \
                          "time frame are identical")
                    # implement here an input if you want to erase the field and rewrite it
                    # we are in a try context so just do something forbidden to rewrite it 
                    # from the underlying suite
                
                # append data to the end
                stacked_field = da.concatenate((expe_field,field), axis=1)
                stacked_t = np.hstack((expe_time, t_axis))
                # time sort the frame
                t_args = np.argsort(stacked_t)
                stacked_field = stacked_field[:,t_args] 
                stacked_t = stacked_t[t_args]
                # save
                experiment[field_name] = {"field" : stacked_field, "time_axis" : stacked_t} 
                print(space, 'field', field_name, "added successfuly (as a suite) to", dict_path)
            except : # if the field is not yet in the dictionnary it is added
                experiment[field_name] = {"field": field,"time_axis": t_axis}
                print(space, 'field', field_name, "added successfuly (for the first time) to", dict_path)
            pickle.dump(experiment,open(dict_path, "wb"))

    @staticmethod
    def memmap_load_chunk(filename, shape, dtype=float, offset=0, slice=None):
        """
        return a view on a given slice of a given file

        filename : str
        shape : tuple
            Total shape of the data in the file
        dtype : 
            numpy dtype of the data in the file
        offset : int
            Skip 'offset' bytes from the beggining of the file
        slice : 
            object that can be used for indexing a numpy array to extract a chunk or
            a subchunk from the file

        ---------------------------------------
        note : if no view can be created memmap returns an ndarray (!!!MEMORY!!!)i
        ---------------------------------------
        """
        data = np.memmap(filename, mode='r', shape=shape, dtype=dtype, offset=offset,order="F")
        if slice !=  None :
            return data[slice]
        else :
            return data

    @staticmethod
    def mmap_phys_field(path, data_files, data_struct):
        """
        path (str) : path to the field directory -> where the field iterations are stored
        data_file (str list) : sorted list of the field iteration files names 
        input shape (list or tuple) : D,S,T,P,N
            D (int) : dimension (number of component) of the field
            S (int): number of domains
            T (int): number of time step
            P (int): number of planes stored in the data file 
            N (list of int) : list of number of mesh point on each domain, e.g 
                              [N in S0, N in S1, N in S3,... ] 
        """
        load = dask.delayed(database.memmap_load_chunk)
        D, S, T, P, N = data_struct 
        
        # get raw view on data files
        field_stack = [da.from_delayed(load(join(path, filename),
                       shape=(database.get_file_length(join(path, filename)),)),
                       shape=(database.get_file_length(join(path,filename)),),
                       dtype=np.float64)  
                       for filename in data_files]
        
        # gather the different domains together 
        bag = [[] for i in range(S)]
        for i, f in enumerate(field_stack):
            it = (i//T) %S
            bag[it].append(da.reshape(f, shape=(P, f.size//P)))
            # careful here, maybe this is not good
        
        temp_field = []
        for i, f in enumerate(bag):
            temp_field.append( da.stack(f).T)
        temp_field = da.vstack(temp_field).T
        # at this point shape is (D*T,P,Ntot)
        field = da.empty(shape=(D,T,P,sum(N)), chunks=(1,1,1,sum(N)))
        for d in range(D):
            for t in range(T):
                field[d,t] = temp_field[T*d + t]
        return field
    
    @staticmethod
    def get_file_list(path):
        return  np.sort([f for f in listdir(path) if isfile(join(path, f))]) 

    @staticmethod
    def list_reynolds(files):
        Re = set()
        all = [i[2:] for f in files for i in f.split("_") if "Re"==i[:2]]
        for e in all:
            Re.add(e)
        Relist = list(Re)
        Re = [[f for f in files if "Re"+re in f.split("_") ] for re in Relist]
        return Relist, Re

    @staticmethod
    def verif_exp(path_interface, dict_name):
        path = join(path_interface,dict_name)
        try:
            d = pickle.load(open(path,"rb"))
            with open(path,"rb") as f:
                d = pickle.load(f) 
                print("Fields will be added in",path)
                print("Already present fields :",np.sort(list(d.keys())))
        
        except FileNotFoundError:
            # create the experiment directory
            os.system("mkdir "+path_interface)
            
            # create the experiment datafile
            with open(path,"wb") as f:
                pickle.dump(dict(), f)   
        return

    @staticmethod
    def get_file_length(path_to_file,float_size=8):
        return os.stat(path_to_file).st_size//float_size

    @staticmethod
    def get_field_info(path, space="phys", float_size = 8):
        """
        path (str) :  path to field directory -> where the fields iterations are stored.
        space (str): can be 'fourier' or 'phys' depending on what data one wants to look at.
        float_size (int) : size in octet, default 8.
        --------------------------------------------------
        """
        all_files = database.get_file_list(path)
        files = np.sort([f for f in all_files if space in f])
        info = files[-1].split("_")
        T = int(info[3].split(".")[0][1:])
        S = int(info[2][1:]) +1
        D = len(files)//T //S
        if D == 1 :
            fieldname = info[1] 
        else :
            fieldname = info[1][:-1]
        
        return files, fieldname, (D,S,T)

    @staticmethod
    def get_axes(global_path, expe_path, axes_files):
        
        # get the time axis
        time = np.loadtxt(join(global_path, "times"))[:,1]
        
        # for later purpose get the number of points in every domain
        vrlist = [np.fromfile(join(global_path, f)) for f in axes_files if "vvrr" in f]
        vN_per_S = [f.size for f in vrlist]
        
        hrlist   = [np.fromfile(join(global_path, f)) for f in axes_files if "Hrr" in f]
        hN_per_S = [f.size for f in hrlist]
        
        vr_axis = None 
        vz_axis = None
        vw_axis = None
        hr_axis = None
        hz_axis = None
        hw_axis = None

        # stack the domain axes to get one axis per dimension
        if vrlist != []:
            vr_axis = np.hstack(vrlist)
            vz_axis = np.hstack([np.fromfile(join(global_path, f)) for f in axes_files if "vvzz" in f])
            vw_axis = np.hstack([np.fromfile(join(global_path, f)) for f in axes_files if "vvweight" in f])
        if hrlist != []:
            hr_axis = np.hstack(hrlist)
            hz_axis = np.hstack([np.fromfile(join(global_path, f)) for f in axes_files if "Hzz" in f])
            hw_axis = np.hstack([np.fromfile(join(global_path, f)) for f in axes_files if "Hweight" in f])   
            
        return (vr_axis, vz_axis, vw_axis), (hr_axis, hz_axis, hw_axis), time, vN_per_S, hN_per_S

    @staticmethod
    def save_axes(sorting_method, path_to_dict, v_axes, h_axes, v_args, h_args):
        experiment = pickle.load(open(path_to_dict, "rb"))
        if v_axes[0] is not None:
            experiment["vr_axis"] = v_axes[0][v_args[sorting_method]] 
            experiment["vz_axis"] = v_axes[1][v_args[sorting_method]] 
            experiment["v_weight"] = v_axes[2][v_args[sorting_method]]
        if h_axes[0] is not None:
            experiment["hr_axis"] = h_axes[0][h_args[sorting_method]] 
            experiment["hz_axis"] = h_axes[1][h_args[sorting_method]] 
            experiment["h_weight"] = h_axes[2][h_args[sorting_method]]

        pickle.dump(experiment, open(path_to_dict, "wb"))

        return