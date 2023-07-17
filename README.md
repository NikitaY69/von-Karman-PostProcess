# Manager for SFEMaNS von Karman flow runs and statistical analysis of turbulence
This package provides a python framework to analyse statistical quantities of 
SFEMaNS von Karman flow runs. 

## Installation
The project should first be cloned with 
```
git clone https://gitlab.lisn.upsaclay.fr/allaglo/von-karman-postprocess.git
```
(the SSH protocol is kind of broken with LISN's gitlab)

### Requirements
You must have a conda (or miniconda) manager in order to install the package. To create an environment dedicated to the project, use the `environment.yaml` configuration file:

```
conda env create -f environment.yaml
```

If you prefer to use your personal python environment, just make sure you possess all the relevent dependencies (cf configuration file).

### Versions
This repo provides two package versions: a cpu one (`vkpp` under branch `main`) and a gpu one (`vkcupp` under branch `gpu`). 

#### Casual installation 
Both versions can simultanuously be installed in your environment. 
To do so, proceed as follow (supposing you just cloned + created environment):

1. activate environment (named statistics if you used the configuration file):
    ```
    cd von-karman-postprocess
    conda activate statistics
    ```
1. install CPU version:
    ```
    pip install .
    ```
1. install GPU version:
    ```
    git checkout gpu
    pip install .
    ```

#### Developer installation
For accustomed git+python users who wish to continue developing the code, keep in mind that your local git repo will **not** sync with the installed package. On top of that, developing in parallel the two versions is impossible. You should indeed not switch branches in a single git repo but instead clone the two branches in <ins>two different</ins> git repos. The procedure is described below. 

1. **In two different directories**, clone each branch:
    ```
    git clone --single-branch --branch <branchname> https://gitlab.lisn.upsaclay.fr/allaglo/von-karman-postprocess.git
    ```
1. Load the environment:
    ```
    conda activate statistics
    ```
1. **In each directory**, you must now install the related package with the edit flag:
    ```
    cd von-karman-postprocess
    pip install -e .
    ```

## Known issues
Because Database creation uses a specific class structure, loading 
files not created with the package will return an error.

### Loading of p_view .pkl files
All relevant functions of the `Database` class being staticmethods, you can import them (just as you used to import them with p_view). Prior to the loading of a database, you should thus add the following lines to your script:
```
from vkpp import Database
memmap_load_chunk = Database.memmap_load_chunk
```

### Loading of old Database (prior to packagisation) .pkl files
Packagisation enabled to distinguish real modules from simple files.
`database.py` was seen as a module without `__init__.py` while it really is not. In order to trick the old .pkl file into thinking that
`database.py` is a real module, you must add the following lines to your script before loading:
```
import sys
sys.path.append('path/to/vkpp/or/vkcupp')
```