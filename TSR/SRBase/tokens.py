import numpy as np

# --------------------- TOKEN DEFAULT VALUES ---------------------
# Default complexity
DEFAULT_COMPLEXITY = 1.
# Default free const int initial value
DEFAULT_FREE_CONST_INIT_VAL = 1.
# Max size for token's name and sympy_repr
MAX_NAME_SIZE = 10
# invalid var_id
INVALID_VAR_ID = 9999999
# invalid func_type
INVALID_FUNC_TYPE = 9999999
# For library : Dummy tokens, n_lengths <= pos < (n_lengths + n_dangling)
DUMMY_TOKEN_NAME = 'dummy'
# For library init : Invalid token name
INVALID_TOKEN_NAME = "-"

# --------------------- TOKEN POSITIONAL PROPERTIES IN PROGRAMS ---------------------
# VectPrograms.append, VectPrograms.update_relationships_pos only work with MAX_NB_CHILDREN = 2
MAX_NB_CHILDREN = 2
# VectPrograms.append, VectPrograms.update_relationships_pos, VectPrograms.get_sibling_idx,
# VectPrograms.get_sibling_idx_of_step prior.RelationshipConstraintPrior get_property_of_relative,
# only work with MAX_NB_SIBLINGS = 1
MAX_NB_SIBLINGS = MAX_NB_CHILDREN - 1
# Max arity value
MAX_ARITY = MAX_NB_CHILDREN
# Out of range tokens, pos >= (n_lengths + n_dangling)
INVALID_TOKEN_NAME = "-"
INVALID_POS   = 9999999
INVALID_DEPTH = 9999999

class Token:
    """
        An object representing a unique mathematical symbol (non_positional & semi_positional), except idx (which
        represents the token's idx in the library and is not encoded here).
        Attributes :
        ----------
        See token.Token.__init__ for full description of parameters.

        name                      :  str (<MAX_NAME_SIZE)
        sympy_repr                :  str (<MAX_NAME_SIZE)
        arity                     :  int
        complexity                :  float
        var_type                  :  int
        function                  :  callable or None
        func_type                 :  int
        init_val                  :  float
        var_id                    :  int
        fixed_const               :  float-like
        is_power                  :  bool
        power                     :  float

        Methods
        -------
        __call__(args)
            Calls the token's function.
    """
    def __init__(self,
                 # ---- Token representation ----
                 name,
                 sympy_repr,
                 # ---- Token main properties ----
                 arity,
                 complexity = DEFAULT_COMPLEXITY,
                 var_type = 0,
                 # Function specific
                 function = None,
                 func_type = 0,
                 # Free constant specific
                 init_val = np.NAN,
                 # Input variable / free constant specific
                 var_id = None,
                 # Fixed constant specific
                 fixed_const = np.NAN,
                 # --- Power specific ---
                 is_power = False,
                 power = np.NAN
                 ):
        """
        Note: __init___ accepts None for some parameters for ease of use which are then converted to the right value and
        type as attributes.
        Parameters
        ----------
        name : str
            A short name for the token (eg. 'add' for addition).
        sympy_repr : str
            Sympy representation of mathematical operation.

        arity : int
            Number of argument of token (eg. 2 for addition, 1 for sinus, 0 for input variables or constants).
            - This token represents a function or a fixed const  (ie. var_type = 0 )      <=> arity >= 0
            - This token represents input_var or free const      (ie. var_type = 1 or 2 ) <=> arity = 0
        complexity : float
            Complexity of token.
        var_type : int
            - If this token represents a function    : var_type = 0 (eg. add, mul, cos, exp).
            - If this token represents an input_var  : var_type = 1 (input variable, eg. x0, x1).
            - If this token represents a free const  : var_type = 2 (free constant,  eg. c0, c1).
            - If this token represents a fixed const : var_type = 3 (eg. pi, 1)
        function : callable or None
            - This token represents a function (ie. var_type = 0 ) <=> this represents the function associated with the
            token. Function of arity = n must be callable using n arguments, each argument consisting in a numpy array
            of floats of shape (int,) or a single float number.
            - This token represents an input_var, a free const or a fixed const (ie. var_type = 1, 2 or 3) <=>
            function = None
        func_type : int or None
            - If this token represents a basic binary function       : func_type = 0 (eg. add, sub, mul, div).
            - If this token represents a trigonometric function      : func_type = 1 (eg. sin, cos, tan).
            - If this token represents a exponential function        : func_type = 2 (eg. log, exp).
            - If this token represents a power function              : func_type = 3 (eg. n2, n3, sqrt, inv).
            - If this token represents a hyperbolic function         : func_type = 4 (eg. sinh, cosh, tanh).
            - If this token represents an arc trigonometric function : func_type = 5 (eg. arcsin, arccos, arctan).
            - If this token represents a sign-based function         : func_type = 6 (eg. neg, abs, sign).
            - If this token represents a custom function             : func_type = 7 (eg. custom).
            - This token does not represent a function               : var_type != 0 <=> func_type == None
            (converted to INVALID_FUNC_ID in __init__)

        init_val : float or np.NAN
            - This token represents a function, a fixed const or an input variable (ie. var_type = 0, 1 or 3)
            <=> init_val = np.NAN
            - This token represents a free const (ie. var_type = 2 )  <=>  init_val = non NaN float
        var_id : int or None
            - This token represents an input_var or a free constant (ie. var_type = 1 or 2) <=> var_id is an integer
            representing the id of the input_var in the dataset or the id of the free const in the free const array.
            - This token represents a function or a fixed constant (ie. var_type = 0 or 3) <=> var_id = None.
            (converted to INVALID_VAR_ID in __init__)
        fixed_const : float or np.NAN
            - This token represents a fixed constant (ie. var_type = 3) <=> fixed_const = non NaN float
            - This token represents a function, an input_var or a free const (ie. var_type = 0, 1 or 2 )
            <=>  fixed_const = non NaN float
        is_power : bool
            True if token is a power token (n2, sqrt, n3 etc.), False else.
        power : float or np.NAN
            - is_power = True <=> power is a float representing the power of a token (0.5 for sqrt, 2 for n2 etc.)
            - is_power = False <=> power is np.NAN

        """
        # ---------------------------- Token representation ----------------------------
        # ---- Assertions ----
        assert isinstance(name,       str),     "name must be a string, %s is not a string" % (str(name))
        assert isinstance(sympy_repr, str),     "sympy_repr must be a string, %s is not a string" % (str(sympy_repr))
        assert len(name      ) < MAX_NAME_SIZE, "name       must be shorter than %d characters, MAX_NAME_SIZE can be changed" % (MAX_NAME_SIZE)
        assert len(sympy_repr) < MAX_NAME_SIZE, "sympy_repr must be shorter than %d characters, MAX_NAME_SIZE can be changed" % (MAX_NAME_SIZE)
        # ---- Attributes ----
        self.name = name                                          # str (<MAX_NAME_SIZE)
        self.sympy_repr = sympy_repr                              # str (<MAX_NAME_SIZE)

        # ---------------------------- Token main properties ----------------------------
        # ---- Assertions ----
        assert isinstance(arity, int), "arity must be an integer, %s is not an integer" % (str(arity))
        assert isinstance(float(complexity), float), "complexity must be castable to a float"
        assert isinstance(int(var_type), int) and int(var_type) <= 3, "var_type must be castable to an integer between 0 and 3"
        assert isinstance(float(fixed_const), float), "fixed_const must be castable to a float"

        # var_type == 1: Token represents input variables (input_var, x0, x1 etc.)
        if var_type == 1:
            assert function is None,            'Token representing input_var (var_type = 1) must have function = None'
            assert arity == 0,                  'Token representing input_var (var_type = 1) must have arity = 0'
            assert isinstance(var_id, int),     'Token representing input_var (var_type = 1) must have var_id is an integer'
            assert np.isnan(init_val),          'Token representing input_var (var_type = 1) must have init_val = np.NAN'
            assert np.isnan(float(fixed_const)),  \
                                                'Token representing input_var (var_type = 1) must have fixed_const = np.NAN'
        # var_type == 0: Token represents a function (add, mul, cos, exp etc.)
        elif var_type == 0:
            assert callable(function),          'Token representing function (var_type = 0) must have a callable function'
            assert arity >= 0,                  'Token representing function (var_type = 0) must have arity >= 0'
            assert var_id is None,              'Token representing function (var_type = 0) must have var_id = None'
            assert np.isnan(init_val),          'Token representing function (var_type = 0) must have init_val = np.NAN'
            assert np.isnan(float(fixed_const)),  \
                                                'Token representing function (var_type = 0) must have fixed_const = np.NAN'
        # var_type == 2: Token represents a free constant (c0, c1 etc.)
        elif var_type == 2:
            assert function is None,            'Token representing free_const (var_type = 2) must have function = None'
            assert arity == 0,                  'Token representing free_const (var_type = 2) must have arity = 0'
            assert isinstance(var_id, int),     'Token representing free_const (var_type = 2) must have var_id is an integer'
            assert isinstance(init_val, float) and not np.isnan(init_val), \
                                                'Token representing free_const (var_type = 2) must have init_val = non NaN float'
            assert np.isnan(float(fixed_const)),  \
                                                'Token representing free_const (var_type = 2) must have fixed_const = np.NAN'
        # var_type == 3: Token represents a fixed constant (e, pi, 1 etc.)
        elif var_type == 3:
            assert function is None,            'Token representing fixed_const (var_type = 3) must have function = None'
            assert arity == 0,                  'Token representing fixed_const (var_type = 3) must have arity = 0'
            assert var_id is None,              'Token representing fixed_const (var_type = 3) must have var_id = None'
            assert np.isnan(init_val),          'Token representing fixed_const (var_type = 3) must have init_val = np.NAN'
            assert not np.isnan(float(fixed_const)),  \
                                                'Token representing fixed_const (var_type = 3) must have fixed_const = non NaN float'
        # ---- Attributes ----
        self.arity = arity                                        # int
        self.complexity = float(complexity)                       # float
        self.var_type = int(var_type)                             # int
        # Function specific
        self.function = function                                  # callable or None
        if self.var_type == 0:
            assert isinstance(int(func_type), int) and int(func_type) <= 7, "func_type must be castable to an integer between 0 and 7"
            self.func_type = int(func_type)                       # int
        else:
            self.func_type = INVALID_FUNC_TYPE                     # int
        # Free constant specific
        self.init_val = init_val                                  # float or np.NAN
        # Input variable / free constant specific
        if self.var_type == 1 or self.var_type == 2:
            self.var_id = int(var_id)                             # int
        else:
            self.var_id = INVALID_VAR_ID                          # int
        # Fixed constant specific
        self.fixed_const = float(fixed_const)                     # float or np.NAN
        # Power specific
        assert isinstance(bool(is_power), bool), 'is_power must be castable to a bool'
        if is_power:
            assert isinstance(power, float) and not np.isnan(power),\
                                                 'Token representing power (is_power = True) must have power = non NaN float'
        else:
            assert np.isnan(power),              'Token not representing power (is_power = False) must have power = np.NAN'
        self.is_power = bool(is_power)                            # bool
        self.power = float(power) if is_power else np.NAN         # float or np.NAN

    def __call__(self, *args):

        # Assert number of args vs arity
        assert len(args) == self.arity, "%i arguments were passed to token %s but token has arity %i" % (len(args), self.name, self.arity)
        if self.var_type == 0:
            return self.function(*args)
        elif self.var_type:
            return self.fixed_const
        
        # Raise error for input_var and free const tokens
        # x0(data_x0, data_x1) would trigger both errors -> use AssertionError for both for simplicity
        else:
            raise AssertionError("Token %s does not represent a function or a fixed constant (var_type=%s), it can not "
                                 "be called."% (self.name, str(self.var_type)))
    # if called in repr() or str(), return name
    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name

class VectTokens:
    """
    Object representing a matrix of positional tokens (positional) ie:
     - non_positional properties: idx + token properties attributes, see token.Token.__init__ for full description.
     - positional properties which are contextual (family relationships, depth etc.).
    This only contains properties expressed as float, int, bool to be jit-able.

    Attributes : In their non-vectorized shapes (types are vectorized)
    ----------
    idx                       : int
        Encodes token's nature, token index in the library.

    ( name                    :  str (<MAX_NAME_SIZE) )
    ( sympy_repr              :  str (<MAX_NAME_SIZE) )
    arity                     :  int
    complexity                :  float
    var_type                  :  int
    func_type                 :  int
    ( function                :  callable or None  )
    ( init_val                  :  float           )
    var_id                    :  int
    ( fixed_const             : float              )
    is_power                  :  bool
    power                     :  float

    pos                      : int
        Position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    pos_batch                : int
        Position in the batch ie in batch dim.
    depth                    : int
        Depth in tree representation of program.
    has_parent_mask          : bool
        True if token has parent, False else.
    has_siblings_mask         : bool
        True if token has at least one sibling, False else.
    has_children_mask         : bool
        True if token has at least one child, False else.
    has_ancestors_mask        : bool
        True if token has at least one ancestor, False else. This is always true for valid tokens as the token itself
        counts as its own ancestor.
    parent_pos               : int
        Parent position in the program ie in time dim (eg. 0 for mul in program = [mul, x0, x1] )
    siblings_pos              : numpy.array of shape (MAX_NB_SIBLINGS,) of int
        Siblings position in the program ie in time dim (eg. 1 for x0 in program = [mul, x0, x1] )
    children_pos              : numpy.array of shape (MAX_NB_CHILDREN,) of int
        Children position in the program ie in time dim (eg. 2 for x1 in program = [mul, x0, x1] )
    ancestors_pos              : numpy.array of shape (shape[1],) of int`
        Ancestors positions in the program ie in time dim counting the token itself as itw own ancestor.
        (eg. [0, 1, 4, 5, INVALID_POS, INVALID_POS] for x1 in program = [mul, add, sin, x0, log, x1]).
    n_siblings                : int
        Number of siblings.
    n_children                : int
        Number of children.
    n_ancestors               : int
        Number of ancestors. This is equal to depth+1 as the token itself counts as its own ancestor.
    """

    def __init__(self, shape, invalid_token_idx):
        """
        Parameters
        ----------
        shape : (int, int)
            Shape of the matrix of tokens.
        invalid_token_idx : int
            Index of the invalid token in the library of tokens
        """

        # ---------------------------- Non-positional properties ----------------------------
        # ---- Shape ----
        assert len(shape) == 2, "Shape of VectTokens object must be 2D."
        # remove line when jit-ing class ?
        self.shape = shape # (int, int)
        self.invalid_token_idx = invalid_token_idx # int

        # ---- Index in library ----
        # Default value
        self.default_idx = self.invalid_token_idx
        # Property
        self.idx = np.full(shape = self.shape, fill_value = self.default_idx, dtype = int)

        # ---- Token representation ----
        # ( name                    :  str (<MAX_NAME_SIZE) )
        # self.tokens_names    = np.full((self.batch_size, self.max_time_step), INVALID_TOKEN_NAME, dtype="S%i"%(Tok.MAX_NAME_SIZE))
        # ( sympy_repr              :  str (<MAX_NAME_SIZE) )

        # ---- Token properties ----
        # Default values
        self.default_arity = 0
        self.default_complexity = DEFAULT_COMPLEXITY
        self.default_var_type = 0
        self.default_var_id = INVALID_VAR_ID
        self.default_func_type = INVALID_FUNC_TYPE
        self.default_is_power = False
        self.default_power    = np.NAN

        # Properties
        self.arity = np.full(shape = self.shape, fill_value = self.default_arity, dtype = int)
        self.complexity = np.full(shape = self.shape, fill_value = self.default_complexity, dtype = float)
        self.var_type = np.full(shape = self.shape, fill_value = self.default_var_type, dtype = int)
        self.var_id = np.full(shape = self.shape, fill_value = self.default_var_id, dtype = int)
        self.func_type = np.full(shape = self.shape, fill_value = self.default_func_type, dtype = int)
        # ( function                :  callable or None  )
        # ( init_val                  :  float           )
        self.is_power = np.full(shape=self.shape, fill_value=self.default_is_power ,  dtype=bool)
        self.power    = np.full(shape=self.shape, fill_value=self.default_power    ,  dtype=float)

        # ---------------------------- Positional properties ----------------------------
        # ---- Position ----
        # Default values
        self.default_pos       = INVALID_POS
        self.default_pos_batch = INVALID_POS
        # Properties : position is the same in all elements of batch
        self.pos               = np.tile(np.arange(0, self.shape[1]), (self.shape[0], 1)).astype(int)
        self.pos_batch         = np.tile(np.arange(0, self.shape[0]), (self.shape[1], 1)).transpose().astype(int)

        # ---- Depth ----
        # Default value
        self.default_depth = INVALID_DEPTH
        # Property
        self.depth = np.full(shape=self.shape, fill_value=self.default_depth, dtype=int )

        # ---- Family relationships ----

        # Token family relationships: family mask
        # Default values
        self.default_has_parent_mask    = False
        self.default_has_siblings_mask  = False
        self.default_has_children_mask  = False
        self.default_has_ancestors_mask = False
        # Properties
        self.has_parent_mask    = np.full(shape=self.shape, fill_value=self.default_has_parent_mask    ,           dtype=bool)
        self.has_siblings_mask  = np.full(shape=self.shape, fill_value=self.default_has_siblings_mask  ,           dtype=bool)
        self.has_children_mask  = np.full(shape=self.shape, fill_value=self.default_has_children_mask  ,           dtype=bool)
        self.has_ancestors_mask = np.full(shape=self.shape, fill_value=self.default_has_ancestors_mask ,           dtype=bool)

        # Token family relationships: pos
        # Default values
        self.default_parent_pos    = INVALID_POS
        self.default_siblings_pos  = INVALID_POS
        self.default_children_pos  = INVALID_POS
        self.default_ancestors_pos = INVALID_POS
        # Properties
        self.parent_pos         = np.full(shape=self.shape,                      fill_value=self.default_parent_pos   , dtype=int)
        self.siblings_pos       = np.full(shape=self.shape + (MAX_NB_SIBLINGS,), fill_value=self.default_siblings_pos , dtype=int)
        self.children_pos       = np.full(shape=self.shape + (MAX_NB_CHILDREN,), fill_value=self.default_children_pos , dtype=int)
        self.ancestors_pos      = np.full(shape=self.shape + (self.shape[1], ),  fill_value=self.default_ancestors_pos, dtype=int)

        # Token family relationships: numbers
        # Default values
        self.default_n_siblings  = 0
        self.default_n_children  = 0
        self.default_n_ancestors = 0
        # Properties
        self.n_siblings         = np.full(shape=self.shape,  fill_value=self.default_n_siblings , dtype=int)
        self.n_children         = np.full(shape=self.shape,  fill_value=self.default_n_children , dtype=int)
        self.n_ancestors        = np.full(shape=self.shape,  fill_value=self.default_n_ancestors, dtype=int)
