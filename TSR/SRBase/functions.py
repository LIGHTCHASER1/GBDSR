import warnings as warnings
import numpy as np
import torch

# Internal imports
import tokens as Tok
from tokens import Token

# Error to raise when a function is unknown
class UnknownFunction(Exception):
    pass

# ---------- Unprotected Functions ----------
def torch_pow(x0, x1):
    if not torch.is_tensor(x0):
        x0 = torch.ones_like(x1) * x0
    return torch.pow(x0, x1)

OPS_UNPROTECTED = [
    # ---- Binary ops ----
    Token (name = 'add', sympy_repr = '+', arity = 2, complexity = 1, var_type = 0, function = torch.add, func_type = 0),
    Token (name = 'sub', sympy_repr = '-', arity = 2, complexity = 1, var_type = 0, function = torch.subtract, func_type = 0),
    Token (name = 'mul', sympy_repr = '*', arity = 2, complexity = 1, var_type = 0, function = torch.multiply, func_type = 0),
    Token (name = 'div', sympy_repr = '/', arity = 2, complexity = 1, var_type = 0, function = torch.divide, func_type = 0),

    # ---- Unary ops ----
    # Trigonometric functions
    Token (name = 'sin', sympy_repr = 'sin', arity = 1, complexity = 1, var_type = 0, function = torch.sin, func_type = 1),
    Token (name = 'cos', sympy_repr = 'cos', arity = 1, complexity = 1, var_type = 0, function = torch.cos, func_type = 1),
    Token (name = 'tan', sympy_repr = 'tan', arity = 1, complexity = 1, var_type = 0, function = torch.tan, func_type = 1),

    # Exponential functions
    Token (name = 'exp', sympy_repr = 'exp', arity = 1, complexity = 1, var_type = 0, function = torch.exp, func_type = 2),
    Token (name = 'log', sympy_repr = 'log', arity = 1, complexity = 1, var_type = 0, function = torch.log, func_type = 2),
    Token (name = 'log1p', sympy_repr = 'log1p', arity = 1, complexity = 1, var_type = 0, function = torch.log1p, func_type = 2),
    Token (name = 'expm1', sympy_repr = 'expm1', arity = 1, complexity = 1, var_type = 0, function = torch.expm1, func_type = 2),

    # Power functions
    Token (name = 'sqrt', sympy_repr = 'sqrt', arity = 1, complexity = 1, var_type = 0, function = torch.sqrt, func_type = 3, is_power = True, power = 0.5),
    Token (name = 'n2'  , sympy_repr = 'n2', arity = 1, complexity = 1, var_type = 0, function = torch.square, func_type = 3, is_power = True, power = 2),
    Token (name = 'n3'  , sympy_repr = 'n3', arity = 1, complexity = 1, var_type = 0, function = lambda x : torch_pow(x, 3), func_type = 3, is_power = True, power = 3),
    Token (name = 'n4'  , sympy_repr = 'n4', arity = 1, complexity = 1, var_type = 0, function = lambda x : torch_pow(x, 4), func_type = 3, is_power = True, power = 4),
    Token (name = "pow" , sympy_repr = "pow"   , arity = 2 , complexity = 1 , var_type = 0, function = torch_pow, func_type = 3, is_power = True),
    Token (name = 'inv', sympy_repr = 'inv', arity = 1, complexity = 1, var_type = 0, function = torch.reciprocal, func_type = 3, is_power = True, power = -1),

    # hyperbolic functions
    Token (name = 'tanh', sympy_repr = 'tanh', arity = 1, complexity = 1, var_type = 0, function = torch.tanh, func_type = 4),
    Token (name = 'sinh', sympy_repr = 'sinh', arity = 1, complexity = 1, var_type = 0, function = torch.sinh, func_type = 4),
    Token (name = 'cosh', sympy_repr = 'cosh', arity = 1, complexity = 1, var_type = 0, function = torch.cosh, func_type = 4),

    # arc functions
    Token (name = 'arcsin', sympy_repr = 'arcsin', arity = 1, complexity = 1, var_type = 0, function = torch.asin, func_type = 5),
    Token (name = 'arccos', sympy_repr = 'arccos', arity = 1, complexity = 1, var_type = 0, function = torch.acos, func_type = 5),
    Token (name = 'arctan', sympy_repr = 'arctan', arity = 1, complexity = 1, var_type = 0, function = torch.atan, func_type = 5),

    # sign-based functions
    Token (name = 'neg', sympy_repr = 'neg', arity = 1, complexity = 1, var_type = 0, function = torch.negative, func_type = 6),
    Token (name = 'abs', sympy_repr = 'abs', arity = 1, complexity = 1, var_type = 0, function = torch.abs, func_type = 6),
    Token (name = 'sign', sympy_repr = 'sign', arity = 1, complexity = 1, var_type = 0, function = torch.sign, func_type = 6),

    # ---- Custom functions ----
    Token (name = 'sigmoid', sympy_repr = 'sigmoid', arity = 1, complexity = 1, var_type = 0, function = torch.sigmoid, func_type = 7),
    Token (name = 'relu', sympy_repr = 'relu', arity = 1, complexity = 1, var_type = 0, function = torch.relu, func_type = 7),
    Token (name = 'erf', sympy_repr = 'erf', arity = 1, complexity = 1, var_type = 0, function = torch.erf, func_type = 7),


]

# ---------- Protected Functions ----------
EPSILON = 0.001
INF = 1e6
EXP_THRESHOLD = float(torch.log(torch.tensor(INF)))

def protected_div(x1, x2):
    # Returns INF if x2 nears zero
    return torch.where(torch.abs(x2) > EPSILON, torch.divide(x1, x2), torch.sign(x1) * torch.sign(x2) * INF)

def protected_exp(x):
    # Returns INF if x nears EXP_THRESHOLD
    return torch.where(torch.abs(x) < EXP_THRESHOLD, torch.exp(x), INF)

def protected_log(x):
    # Returns INF if x nears zero
    return torch.where(x > EPSILON, torch.log(x), - INF)

def protected_sqrt(x):
    # Avoid taking a square of a negative number
    return torch.where(x >= 0, torch.sqrt(x), 0.)

def protected_inv(x):
    # Returns INF if x nears zero
    return torch.where(torch.abs(x) > EPSILON, torch.reciprocal(x), torch.sign(x) * INF)

def protected_n2(x1):
    npow = 2
    sign = 1 if int(npow)%2 == 0 else torch.sign(x1) # Takes the sign of x1 if power is odd
    # Caps square to avoid overflow, returns infinity
    return torch.where(torch.abs(x1) < INF, torch.pow(x1, npow), sign * INF)

def protected_n3(x1):
    npow = 3
    sign = 1 if int(npow)%2 == 0 else torch.sign(x1) # Takes the sign of x1 if power is odd
    # Caps square to avoid overflow, returns infinity
    return torch.where(torch.abs(x1) < INF, torch.pow(x1, npow), sign * INF)

def protected_n4(x1):
    npow = 4
    sign = 1 if int(npow)%2 == 0 else torch.sign(x1) # Takes the sign of x1 if power is odd
    # Caps square to avoid overflow, returns infinity
    return torch.where(torch.abs(x1) < INF, torch.pow(x1, npow), sign * INF)

def protected_arcsin (x1):
    # Handles arcsin, returns infinity with the sign of x1 for values outside the domain
    return torch.where(torch.abs(x1) < 1 - EPSILON, torch.arcsin(x1), torch.sign(x1) * INF)

def protected_arccos (x1):
    # Handles arccos, returns infinity with the sign of x1 for values outside the domain
    return torch.where(torch.abs(x1) < 1 - EPSILON, torch.arccos(x1), torch.sign(x1) * INF)

def protected_torch_pow(x0, x1):
    # Handles power function, caps at positive/negative infinity to avoid overflow
    if not torch.is_tensor(x0):
        x0 = torch.ones_like(x1) * x0

    # Handle negative bases with non-integer exponents
    result_is_nan = torch.isnan(torch.pow(x0, x1))
    x0 = torch.where(result_is_nan, torch.abs(x0), x0)

    y = torch.pow(x0, x1)
    # Handle overflow
    y = torch.where(torch.abs(y) < INF, y, torch.sign(y) * INF)
    # Handle underflow
    y = torch.where(torch.abs(y) > EPSILON, y, 0.)
    return y

OPS_PROTECTED = [
    # Binary operations
    Token (name = "div"    , sympy_repr = "/"      , arity = 2 , complexity = 1 , var_type = 0, function = protected_div    , func_type = 0),
    # Unary operations
    Token (name = "exp"    , sympy_repr = "exp"    , arity = 1 , complexity = 1 , var_type = 0, function = protected_exp    , func_type = 2),
    Token (name = "log"    , sympy_repr = "log"    , arity = 1 , complexity = 1 , var_type = 0, function = protected_log    , func_type = 2),
    Token (name = "sqrt"   , sympy_repr = "sqrt"   , arity = 1 , complexity = 1 , var_type = 0, function = protected_sqrt   , func_type = 3, is_power = True, power = 0.5),
    Token (name = "n2"     , sympy_repr = "n2"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n2     , func_type = 3, is_power = True, power = 2),
    Token (name = "inv"    , sympy_repr = "1/"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_inv    , func_type = 3, is_power = True, power = -1),
    Token (name = "arccos" , sympy_repr = "arccos" , arity = 1 , complexity = 1 , var_type = 0, function = protected_arccos , func_type = 5),
    Token (name = "arcsin" , sympy_repr = "arcsin" , arity = 1 , complexity = 1 , var_type = 0, function = protected_arcsin , func_type = 5),

    Token (name = "n3"     , sympy_repr = "n3"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n3     ,func_type = 3, is_power = True, power = 3),
    Token (name = "n4"     , sympy_repr = "n4"     , arity = 1 , complexity = 1 , var_type = 0, function = protected_n4     ,func_type = 3, is_power = True, power = 4),

    # Custom binary operations
    Token (name = "pow"     , sympy_repr = "pow"   , arity = 2 , complexity = 1 , var_type = 0, function = protected_torch_pow, func_type = 3, is_power = True),

]

# ------------- protected functions -------------

OPS_UNPROTECTED_DICT = {op.name: op for op in OPS_UNPROTECTED}
# Copy unprotected operations
OPS_PROTECTED_DICT = OPS_UNPROTECTED_DICT.copy()
# Update protected operations when defined
OPS_PROTECTED_DICT.update( {op.name: op for op in OPS_PROTECTED} )

# -------------------------------------------- Make Tokens --------------------------------------------
# ----Utils Functions ----
def retrieve_complexity(complexity_dict, curr_name):
    """
    Helper function to safely retrieve complexity of token named curr_name from a dictionary of complexities
    (complexity_dict).
    Parameters
    ----------
    complexity_dict : dict of {str : float} or None
        If dictionary is None, returns token.DEFAULT_COMPLEXITY.
    curr_name : str
        If curr_name is not in units_dict keys, returns token.DEFAULT_COMPLEXITY.
    Returns
    -------
    curr_complexity : float
        Complexity of token.
    """
    curr_complexity = Tok.DEFAULT_COMPLEXITY
    if complexity_dict is not None:
        try:
            curr_complexity = complexity_dict[curr_name]
        except KeyError:
            warnings.warn(
                "Compiexity of token %s not found in complexity dictionary %s, using complexity = %f" % (curr_name, complexity_dict, curr_complexity)
            )
    curr_complexity = float(curr_complexity)
    return curr_complexity
def retrieve_init_val (init_val_dict, curr_name):
    curr_init_val = Tok.DEFAULT_FREE_CONST_INIT_VAL
    if init_val_dict is not None:
        try:
            curr_init_val = init_val_dict[curr_name]
        except KeyError:
            warnings.warn(
                "Initial value of token %s not found in initial value dictionary %s, using initial value = %f" %
                (curr_name, init_val_dict, curr_init_val))
    curr_init_val = float(curr_init_val)
    return curr_init_val

def make_tokens(
        # operations
        op_names = 'all',
        use_protected_ops = False,
        # input variables
        input_var_ids = None,
        input_var_complexity = None,
        # constants
        constants = None,
        constants_complexity = None,
        # free constants
        free_constants = None,
        free_constants_init_val = None,
        free_constants_complexity = None,
        ):
    """
        Makes a list of tokens for a run based on a list of operation names, input variables ids and constants values.
        Parameters
        ----------
        -------- operations --------
        op_names : list of str or str, optional
            List of names of operations that will be used for a run (eg. ["mul", "add", "neg", "inv", "sin"]), or "all"
            to use all available tokens. By default, op_names = "all".
        use_protected_ops : bool, optional
            If True safe functions defined in functions.OPS_PROTECTED_DICT in place when available (eg. sqrt(abs(x))
            instead of sqrt(x)). False by default.
        -------- input variables --------
        input_var_ids : dict of { str : int } or None, optional
            Dictionary containing input variables names as keys (eg. 'x', 'v', 't') and corresponding input variables
            ids in dataset (eg. 0, 1, 2). None if no input variables to create. None by default.
        input_var_complexity : dict of { str : float } or None, optional
            Dictionary containing input variables names as keys (eg. 'x', 'v', 't') and corresponding complexities
            (eg. 0., 1., 0.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        -------- constants --------
        constants : dict of { str : float } or None, optional
            Dictionary containing constant names as keys (eg. 'pi', 'c', 'M') and corresponding float values
            (eg. np.pi, 3e8, 1e6). None if no constants to create. None by default.
        constants_complexity : dict of { str : float } or None, optional
            Dictionary containing constants names as keys (eg. 'pi', 'c', 'M') and corresponding complexities
            (eg. 0., 0., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        -------- free constants --------
        free_constants : set of { str } or None, optional
            Set containing free constant names (eg. 'c0', 'c1', 'c2'). None if no free constants to create.
            None by default.
        free_constants_init_val : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding float initial
            values to use during optimization process (eg. 1., 1., 1.). None will result in the usage of
            token.DEFAULT_FREE_CONST_INIT_VAL as initial values. None by default.
        free_constants_complexity : dict of { str : float } or None, optional
            Dictionary containing free constants names as keys (eg. 'c0', 'c1', 'c2') and corresponding complexities
            (eg. 1., 1., 1.). If None, complexity = token.DEFAULT_COMPLEXITY will be encoded to tokens. None by default.
        Returns
        -------
        list of token.Token
            List of tokens used for this run.

        Examples
        -------
            my_tokens = make_tokens(
                # operations
                op_names             = ["mul", "add", "neg", "inv", "sin"],
                use_protected_ops    = False,
                # input variables
                input_var_ids        = {"x" : 0         , "v" : 1          , "t" : 2,        },
                input_var_complexity = {"x" : 0.        , "v" : 1.         , "t" : 0.,       },
                # constants
                constants            = {"pi" : np.pi     , "c" : 3e8       , "M" : 1e6       },
                constants_complexity = {"pi" : 0.        , "c" : 0.        , "M" : 1.        },
                                    )
    """
    # ---- Operations ----
    tokens_ops = []
    # Use protected functions or not
    ops_dict = OPS_PROTECTED_DICT if use_protected_ops else OPS_UNPROTECTED_DICT
    # Using all available tokens
    if op_names == 'all':
        tokens_ops = list(ops_dict.values())
    # Append desired functions tokens
    else:
        for name in op_names:
            try:
                tokens_ops.append(ops_dict[name])
            except KeyError:
                raise UnknownFunction("%s is unknown, define a custom function in functions.py or use a known one listed in %s" % (name, ops_dict))
    # ---- Input variables ----
    tokens_input_vars = []
    if input_var_ids is not None:
        for var_name, var_id in input_var_ids.items():
            # Complexity
            complexity = retrieve_complexity(complexity_dict = input_var_complexity, curr_name = var_name)
            # Token creation
            tokens_input_vars.append(Token(name = var_name,
                                          sympy_repr = var_name,
                                          arity = 0,
                                          complexity = complexity,
                                          var_type = 1,
                                          # Input variable specific
                                          var_id = var_id,
                                          ))
    # ---- Fixed Constants ----
    tokens_constants = []
    if constants is not None:
        # Interating through constants
        for const_name, const_val in constants.items():
            # Complexity
            complexity = retrieve_complexity(complexity_dict = constants_complexity, curr_name = const_name)
            # Token creation
            # Very important to put const as a default arg of lambda function
            # https://stackoverflow.com/questions/19837486/lambda-in-a-loop
            # or use def MakeConstFunc(x): return lambda: x
            tokens_constants.append(Token(name = const_name,
                                          sympy_repr = const_name,
                                          arity = 0,
                                          complexity = complexity,
                                          var_type = 3,
                                          # Constant specific
                                          fixed_const = const_val,
                                          ))
    # ---- Free constants ----
    tokens_free_constants = []
    if free_constants is not None:
        free_constants_sorted = list(sorted(free_constants))
        # enumerating on sorted list rather than set
        # Iterating through free constants
        for i, free_const_name in enumerate(free_constants_sorted):
            # Complexity
            complexity = retrieve_complexity(complexity_dict = free_constants_complexity, curr_name = free_const_name)
            # Initial value
            init_val = retrieve_init_val(init_val_dict = free_constants_init_val, curr_name = free_const_name)
            # Token creation
            tokens_free_constants.append(Token(name = free_const_name,
                                               sympy_repr = free_const_name,
                                               arity = 0,
                                               complexity = complexity,
                                               var_type = 2,
                                               # Free constant specific
                                               var_id = i,
                                               init_val = init_val,
                                               ))
    return np.array(tokens_ops + tokens_input_vars + tokens_constants + tokens_free_constants)

# main for testing
if __name__ == '__main__':
    # print(len(OPS_UNPROTECTED_DICT))
    # print(len(OPS_PROTECTED_DICT))
    my_tokens = make_tokens(
    # operations
    op_names             = ["mul", "add", "neg", "inv", "sin"],
    use_protected_ops    = False,
    # input variables
    input_var_ids        = {"x0" : 0, "x1" : 1          , "x2" : 2,        },
    input_var_complexity = {"x0" : 1.        , "x1" : 1.         , "x2" : 1.,       },
    # constants
    constants            = {"pi" : np.pi     , "c" : 3e8       , "E" : np.e       },
    constants_complexity = {"pi" : 0.        , "c" : 0.        , "E" : 0.        },
    # free constants
    free_constants       = {"c0", "c1", "c2"},
    free_constants_init_val = {"c0" : 1.       , "c1" : 1.        , "c2" : 1.       },
                        )
    print(my_tokens)