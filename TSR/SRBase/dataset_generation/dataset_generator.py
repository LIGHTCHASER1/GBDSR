import warnings as warnings
import numpy as np
import torch
import random as rd

# To append the path of TSR/SRBase to sys.path in order to successfully import the internal modules
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Internal imports
import tokens as Tok
import library as Lib
import functions as Func
import execute as Exec
from reader import get_dataset_configuration

# Default values
DEFAULT_MAX_PREFIX_LEN = 20
DEFAULT_MAX_OPS = 5
DEFAULT_MAX_VARIABLES_NUM = 2
DEFAULT_VARIABLES_RANGE = [-10, 10]
DEFAULT_SCATTERS_NUM = 100
DEFAULT_MAX_FREE_CONSTANTS_NUM = 3
DEFAULT_FREE_CONSTANTS_RANGE = [-2, 2]
DEFAULT_MAX_CONSTANTS_NUM = 2

# ---- Library configuration ----
def set_library_configuration():
    '''
    Special settings for library configuration.
    Specifically, use_protected_ops/fixed constant's (var_type == 3) value and complexity should be set in args_make_tokens.
    '''
    args_library_spec = {}
    custom_tokens = []
    args_make_tokens = {
        # operations
        "op_names"             : None,
        "use_protected_ops"    : False,
        # input variables
        "input_var_ids"        : None,
        "input_var_complexity" : None,
        # constants
        "constants"            : {"pi" : np.pi     , "E" : np.e             },
        "constants_complexity" : {"pi" : 0.        , "E" : 0.               },
    }
    superparent_name = "y"

    args_library_spec['custom_tokens'] = custom_tokens
    args_library_spec['args_make_tokens'] = args_make_tokens
    args_library_spec['superparent_name'] = superparent_name

    return args_library_spec


# ---- Dataset generation ----
def create_DatasetGenerator():
    '''
    Create a DatasetGenerator object using the configuration file.
    '''
    args_generate_datasets = get_dataset_configuration()
    args_library = set_library_configuration()
    args_generate_datasets.update(args_library)
    args_generator = args_generate_datasets
    return DatasetGenerator(args_generator = args_generator)

class DatasetGenerator:
    '''
    An object to generate a dataset using the configuration file.
    Parameters of args_generate_datsets
    ----------
    max_prefix_len : int
    max_ops        : int
    max_variables_num : int
    variables_range : list
    scatters_num : int
    max_free_constants_num : int
    free_constants_range : list
    max_constants_num : int
    operation_prob : dict

    Parameters of args_library_spec
    ----------
    custom_tokens : list
    args_make_tokens : dict
    superparent_name : str
    '''
    def __init__(self,
                 args_generator = None):
        '''
        An object to generate a dataset using the configuration file.
        args_generator : None or dict
            consists of args_generate_datasets and args_library_spec.
        Parameters of args_generate_datsets : None or dict object
        ----------
        max_prefix_len : int
            Maximum length of the prefix tokens.
        max_ops        : int
            Maximum number of operations.
        max_variables_num : int
            Maximum number of variables(var_type == 1).
        variables_range : list
            Range of variables.
        scatters_num : int
            Number of scatters of (X, y) will be created.
        max_free_constants_num : int
            Maximum number of free constants (var_type == 2).
        free_constants_range : list
            Range of free constants.
        max_constants_num : int
            Maximum number of constants (var_type == 3).
        operation_prob : dict
            Probability of each operation for generating SR programs.

        Parameters of args_library_spec : None or dict object
        ----------
        custom_tokens : list
            List of custom tokens.
            for eg:
                add = Tok.Token(name='add', sympy_repr='add', arity=2, complexity=0, var_type=0,
                        function=np.add, func_type=7,
                        var_id=None)
                custom_tokens = [add,]
        args_make_tokens : dict
            Arguments for make_tokens function.
            Specially, use_protected_ops/fixed constant's (var_type == 3) value and complexity should be set in this argument.
            However, 
            for eg: 
                args_make_tokens = {
                # operations
                ("op_names"             : list of str, will be set according to next argument),
                "use_protected_ops"     : False,
                # input variables
                ("input_var_ids"        : dict, will be set according to next argument),
                ("input_var_complexity" : dict, will be set according to next argument),
                # free constants
                ("free_constants"          : list of str, will be set according to next argument),
                ("free_constants_init_val" : dict, will be set according to next argument),
                # constants
                "constants"            : {"pi" : np.pi     , "E" : np.e             },
                "constants_complexity" : {"pi" : 0.        , "E" : 0.               },
                            }
        superparent_name : str
            Name of the superparent.
            for eg:
                superparent_name = "y"
        '''
        # ---------------------------------------- PARAMETERS ----------------------------------------
        # ---- Basic dataset generation parameters ----

        # Unary-binary tree specific
        self.max_len = args_generator['max_len']
        self.max_ops = args_generator['max_ops']

        # Variables specific (var_type == 1)
        self.max_variables_num = args_generator['max_variables_num']
        self.variables_range = args_generator['variables_range']
        self.scatters_num = args_generator['scatters_num']

        # Free constants specific (var_type == 2)
        self.max_free_constants_num = args_generator['max_free_constants_num']
        self.free_constants_range = args_generator['free_constants_range']

        # Fixed constants specific (var_type == 3)
        self.max_constants_num = args_generator['max_constants_num']

        # Operations specific (var_type == 0)
        self.operation_prob = args_generator['operation_prob']

        # ---- Basic library parameters ----
        self.custom_tokens = args_generator['custom_tokens']
        # Assert args_make_tokens has valid values of use_protected_ops/fixed constant's (var_type == 3) value and complexity
        self.args_make_tokens = args_generator['args_make_tokens']
        self.superparent_name = args_generator['superparent_name']

        # ---------------------------------------- ATTRIBUTES ----------------------------------------
        self.library = self.gen_library()
        self.SE = Exec.SkeletonExecutor(self.library,
                                        self.max_len,
                                        self.max_ops,
                                        self.operation_prob)
    
    def gen_library(self):
        '''
        Generate a library using the configuration file.
        '''
        # ---- Basic library parameters ----
        custom_tokens = self.custom_tokens
        args_make_tokens = self.args_make_tokens
        superparent_name = self.superparent_name

        # ---- Generate args_make_tokens ----
        '''
        for eg: 
        args_make_tokens = {
        # operations
        ("op_names"                : list of str, will be set in this argument),
        "use_protected_ops"        : False,
        # input variables
        ("input_var_ids"           : dict, will be set in this argument),
        ("input_var_complexity"    : dict, will be set in this argument),
        # free constants
        ("free_constants"          : list of str, will be set in this argument),
        ("free_constants_init_val" : dict, will be set in this argument),
        # constants
        "constants"            : {"pi" : np.pi     , "E" : np.e             },
        "constants_complexity" : {"pi" : 0.        , "E" : 0.               },
                    }
        '''
        # operation names
        args_make_tokens['op_names'] = list(self.operation_prob.keys())
        # input variables
        variables_num = rd.randint(1, self.max_variables_num)
        args_make_tokens['input_var_ids'] = {f'x{i}' : i - 1 for i in range(1, variables_num + 1)}
        args_make_tokens['input_var_complexity'] = {f'x{i}' : 1. for i in range(1, variables_num + 1)}
        # free constants
        free_constants_num = rd.randint(1, self.max_free_constants_num)
        args_make_tokens['free_constants'] = [f'c{i}' for i in range(1, free_constants_num + 1)]
        args_make_tokens['free_constants_init_val'] = \
            {f'c{i}' : rd.uniform(*self.free_constants_range) for i in range(1, free_constants_num + 1)}
        args_make_tokens['free_constants_complexity'] = \
            {f'c{i}' : 0. for i in range(1, free_constants_num + 1)}
        # ---- Generate library ----
        library = Lib.Library(custom_tokens = custom_tokens,
                              args_make_tokens = args_make_tokens,
                              superparent_name = superparent_name)
        return library
    
    def Generation(self):
        '''
        Generate a dataset using the configuration file.
        '''
        return self.SE.generate_expression(nb_total_ops = 5)




if __name__ == '__main__':
    DG = create_DatasetGenerator()
    # print(DG.library)
    print(DG.Generation())