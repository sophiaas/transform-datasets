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
    
