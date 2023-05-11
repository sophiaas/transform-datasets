import numpy as np
from itertools import product as cproduct


class Group:
    """General group of size n acting on vectors of length N"""

    def __init__(self, n=1, N=1):
        self.N = N
        self.n = n

    def act(self, g, x):  # trivial action
        return x


class Element:
    """General element in a group"""

    def __init__(self, group, normal_form=None):
        self.group = group
        if normal_form is not None:
            self.normal_form = normal_form
        else:
            self.normal_form = []

    def __call__(self, x):
        return self.group.act(self, x)


class Commutative(Group):
    """Commutative group G of type [m1, m2, ..., ml]:
        G = Z/m1Z x Z/m2Z x ... x Z/mlZ
    |G| = m1 * m2 * ... * ml
    """

    def __init__(self, M=[2, 3]):
        self.M = M
        self.n = np.product(M)  # order of group
        self.l = len(M)  # number of products

    def multiply(self, g, h):
        """ returns g*h in the group as a tuple [a1, a2, ..., al] """
        return [np.mod(g[i] + h[i], m) for i, m in enumerate(self.M)]

    def index_of_element(self, g):
        """Returns the index of the group element g
        Note:  Can be made much faster by analytically finding index"""

        return list(cproduct(*[range(a) for a in self.M])).index(tuple(g))

    def element_at_index(self, idx):
        """ Slow """
        return list(cproduct(*[range(a) for a in self.M]))[idx]

    def CayleyTable(self):
        """Returns the nxn multiplication table for G
        Note: ordering given by itertools cartesian product"""

        CT = np.zeros((self.n, self.n))
        for i, g in enumerate(cproduct(*[range(a) for a in self.M])):
            for j, h in enumerate(cproduct(*[range(b) for b in self.M])):
                CT[i, j] = self.index_of_element(self.multiply(g, h))
        CT = CT.astype('int')
        return CT

    def act(self, g, x):
        """default action is on vectors x = [x_0, x_1, ..., x_n]
        where the ordering is the same as the group (Cartesian product)"""
        y = np.zeros(len(x))
        for i in range(self.n):
            y[self.index_of_element(self.multiply(g, self.element_at_index(i)))] = x[i]
        return y

    def rand_element(self, x=None, return_element=False):
        g = [np.random.randint(self.M[i]) for i in range(self.l)]
        if x is None:
            return g
        if return_element:
            return self.act(g, x), g
        return self.act(g, x)
    

    
# Example Hierarchical Swap/Reflection Groups acting on R^N, N = 2^k
# C.Hillar, 2020, Awecom Inc

"""
NON-COMMUTATIVE -- not yet converted into transform_datasets format
"""

class Reflection(Group):
    """ Hierarchical Reflection group acting on vectors of length N = 2^k
    """
    @staticmethod
    def rand_normal_form(k):
        normal_form = [[] for i in range(k)]
        for i in range(0, k):
            normal_form[i] = list((np.random.random(2 ** i) > .5).astype(int))
        return normal_form

    def __init__(self, k):
        super().__init__(n=2**(2**k-1), N=2**k)
        self.k = k

    def generator(self, index=[0]):
        """ returns generator indexed by binary vector of length 2^l 
            e.g. g_{\empty} is index [0], g_{1} is index = [0, 1], g_{01}*g_{11} is index = [0, 1, 0, 1]
        """
        normal_form = [[0] for i in range(self.k)]
        for i in range(1, self.k):
            normal_form[i] = [0] * (i + 1)
        normal_form[len(index)- 1] = index
        new_element = Element(self, normal_form)
        new_element.index = index
        return new_element

    def element(self, normal_form=None):
        if normal_form is None:
            return self.rand_element()
        return Element(self, normal_form=normal_form)

    def basic_op(self, x):
        return x[::-1]  # reflect action

    def act_generator(self, index, x):
        """ acts by element of corresponding normal form product group 
        """
        size = x.shape[0] // len(index)
        for i, op in enumerate(index):
            if op == 1:
                x[size * i:size * (i+1)] = self.basic_op(x[size * i:size * (i+1)])
        return x

    def act(self, g, x, opposite=True):
        """ action of group element g on vector x
            opposite = True if action using normal form goes from global generators to local
        """
        output = x.copy()
        if opposite is True:  # normal form from global to local (action left to right)
            arr = g.normal_form
        else:  # normal form from local to global (action right to left)
            arr = g.normal_form[::-1]
        for index in arr:
            output = self.act_generator(index, output)
        return output

    def rand_element(self, x=None):
        g = Element(self, normal_form=self.rand_normal_form(self.k))
        if x is None:
            return g
        return self.act(g, x)

class Swap(Reflection):
    """ Hierarchical Swap group acting on vectors of length N
        [Note: this group is isomorphic to Reflection]
    """
    def basic_op(self, x):
        size = x.shape[0]
        return np.hstack([x[size//2:], x[:size//2]])  # swap action

class DataFactory():
    """ Makes data using Groups """
    def __init__(self, group):
        self.group = group

    def random_actions(self, x, m):
        """ x is initial vector to act randomly on, m is number of samples """
        output = np.zeros((m, x.shape[0]))
        for i in range(m):
            output[i] = self.group.rand_element(x)
        return output