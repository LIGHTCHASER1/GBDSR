import warnings

import numpy as np
import random as rd
import numpy.random as rng
import torch as torch

# ---------------------------- SINGLE EXECUTION ----------------------------
### ATTENTION : Default program tokens are in reverse Polish notation order (suffix order) ###
def ExecuteProgram(input_var_data, program_tokens, free_const_values = None):
    """
    Executes a symbolic function program.
    Parameters
    ----------
    input_var_data : torch.tensor of shape (n_dim, ?,) of float
        Values of the input variables of the problem with n_dim = nb of input variables.
    program_tokens : list of token.Token
        Symbolic function program in reverse Polish notation order (suffix order).
    free_const_values : torch.tensor of shape (n_free_const,) of float or None
        Current values of free constants with for program made of program_tokens n_free_const = nb of choosable free
        constants (library.n_free_constants). free_const_values must be given if program_tokens contains one or more
         free const tokens.
    Returns
    -------
    y : torch.tensor of shape (?,) of float
        Result of computation.
    """

    # Size
    (n_dim, data_size,) = input_var_data.shape

    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range(start, -1, -1):
        # Current token
        curr_token = program_tokens[i]
        # Current token is a variable or constant
        if curr_token.arity == 0:
            # Function type token
            if curr_token.var_type == 0:
                raise ValueError("Function type (var_type == 0) token cannot be a variable or constant (arity == 0).")
            # Input variable token (eg. x1, x2 etc.)
            elif curr_token.var_type == 1:
                curr_stack.append(input_var_data[curr_token.var_id])
            # Free constant token (eg. c1, c2 etc.)
            elif curr_token.var_type == 2:
                if free_const_values is not None:
                    curr_stack.append(free_const_values[curr_token.var_id])
                else:
                    raise ValueError("Free constant token (var_type == 2) found but no free_const_values given.")
            # Fixed constant token (eg. pi, e etc.)
            elif curr_token.var_type == 3:
                curr_stack.append(curr_token.value)
            else:
                raise NotImplementedError("Unknown var_type value %d encountered in ExecuteProgram of execute.py." % curr_token.var_type)
        # Non-terminal token
        elif curr_token.arity > 0:
            # Last pending elements are those needed for next computation (in reverse order)
            args = curr_stack[-curr_token.arity:][::-1]
            res = curr_token.function(*args)
            # Removing those pending elements as they are used
            curr_stack = curr_stack[:-curr_token.arity]
            # Appending last result to stack
            curr_stack.append(res)
    y = curr_stack[0]
    return y

def ComputeInfixNotation (program_tokens):
    """
    Computes infix str representation of a program.
    (which is the usual way to note symbolic function: +34 (in polish notation) = 3+4 (in infix notation))
    Parameters
    ----------
    program_tokens : list of token.Token
        List of tokens making up the program.
    Returns
    -------
    program_str : str
    """
    # Number of tokens in the program
    n_tokens = len(program_tokens)

    # Current stack of computed results
    curr_stack = []

    # De-stacking program (iterating from last token to first)
    start = n_tokens - 1
    for i in range(start, -1, -1):
        curr_token = program_tokens[i]
        # Last pending elements are those needed for next computation (in reverse order)
        args = curr_stack[-curr_token.arity:][::-1]
        if curr_token.arity == 0:
            res = curr_token.sympy_repr
        elif curr_token.arity == 1:
            if curr_token.is_power is True:
                pow = '{:g}'.format(curr_token.power) # without trailing zeros
                res = "((%s)**(%s))" % (args[0], pow)
            else:
                res = "(%s(%s))" % (curr_token.sympy_repr, args[0])
        elif curr_token.arity == 2:
            res = "(%s%s%s)" % (args[0], curr_token.sympy_repr, args[1])
        elif curr_token.arity > 2:
            args_str = ""
            for arg in args:
                args_str += "%s," % arg
            args_str = args_str[:-1] # remove last comma
            res = "(%s(%s))" % (curr_token.sympy_repr, args_str)
        curr_stack = curr_stack[:-curr_token.arity]
        curr_stack.append(res)
    return curr_stack[0]


class SkeletonExecutor:
    def __init__(self,
                 library,
                 max_len,
                 max_ops,
                 operation_prob):
        # basic parameters
        self.library = library
        self.max_len = max_len
        self.max_ops = max_ops
        self.operation_prob = operation_prob
        # arity index for operations in library
        '''
        eg. [mul, sin, cos, add] 
                        => [2, 1, 1, 2]
        '''
        self.library_invalid_idx = self.library.invalid_idx
        # superparent, dummy, invalid are EXCLUDED
        # only include non-placeholders
        self.library_tokens_arity = self.library.get_choosable_prop("arity")
        # sub-library for arity == 0 : list
        self.operands = self.get_sub_arity_library_prob_dict(0)
        # Free consts (var_type == 2) usually in the end of operands
        # Shuffling the operands list to make the free consts to be randomly chosen
        rd.shuffle(self.operands)
        self.pos_dict = {element: index for index, element in enumerate(self.operands)}
        # sub-library for arity == 1 : list
        # self.una_ops is a list of Tokens, self.una_prob is a list of probabilities
        self.una_ops, self.una_prob = self.get_sub_arity_library_prob_dict(1)
        # sub-library for arity == 2 : list
        # self.bin_ops is a list of Tokens, self.bin_prob is a list of probabilities
        self.bin_ops, self.bin_prob = self.get_sub_arity_library_prob_dict(2)

        self.all_ops = self.una_ops + self.bin_ops
        
        # probability normalization

        self.una_prob = [i / sum(self.una_prob) for i in self.una_prob]
        self.bin_prob = [i / sum(self.bin_prob) for i in self.bin_prob]

        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist()
        self.ubi_dist = self.generate_ubi_dist()
    
    def gen_invalid_token(self):
        # generate invalid token for unary-binary tree
        # invalid token is a Token object
        return self.library[self.library_invalid_idx]

    def get_sub_arity_library_prob_dict(self, spe_arity):
        # get sub library and probability information dict for given arity
        sub_library = [self.library[i] for i, value in enumerate(self.library_tokens_arity) if value == spe_arity]
        #sub_prob = [self.operation_prob[self.library[i]] for i, value in enumerate(self.library_arity) if value == spe_arity]
        if spe_arity == 1 or spe_arity == 2:
            sub_prob = [self.operation_prob[self.library[i].name] for i, value in enumerate(self.library_tokens_arity) if value == spe_arity]
            return sub_library, sub_prob
        else:
            return sub_library


    def generate_bin_dist(self):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]  # [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796]
        for i in range(1, 2 * self.max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(self.max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * self.max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D
    
    def generate_ubi_dist(self):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * self.max_ops + 1)]))
        for n in range(1, 2 * self.max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * self.max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D
    
    def sample_next_pos_ubi(self, nb_empty, nb_ops):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity
    
    def get_leaf(self, curr_leaves):
        """
        Get a list of leaves for the current node.
        """
        if curr_leaves:
            max_idxs = max([self.pos_dict[i] for i in curr_leaves]) + 1
        else:
            max_idxs = 0
        return [self.operands[rng.randint(low = 0, high = min(max_idxs + 1, len(self.operands)))]]


    def generate_expression(self, nb_total_ops):
        """
        Create a tree with exactly `nb_total_ops` operators.
        Return a prefix expression stack.
        """
        stack = [self.gen_invalid_token()]
        nb_empty = 1 # number of empty nodes
        l_leaves = 0 # left leaves (number of leaves on the left of the current node)
        t_leaves = 1 # total number of leaves (just for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops)
            if arity == 1:
                # op is a Token object
                op = rng.choice(self.una_ops, p=self.una_prob)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_prob)
            nb_empty += op.arity - 1 - skipped # created empty nodes - skipped future leaves
            t_leaves += op.arity - 1 # update number of total leaves
            l_leaves += skipped # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v == self.gen_invalid_token()][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [self.gen_invalid_token() for _ in range(op.arity)]
                + stack[pos + 1:]
            )
        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is self.gen_invalid_token()]) == t_leaves

        leaves = []
        curr_leaves = set()
        # create leaves
        for _ in range(t_leaves):
            new_element = self.get_leaf(curr_leaves)
            leaves.append(new_element)
            curr_leaves.add(*new_element)

        # replace invalid tokens by leaves
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is self.gen_invalid_token():
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0, "Not all leaves were used in the expression."
        return stack




