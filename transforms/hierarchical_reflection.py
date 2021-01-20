# Example Hierarchical Swap/Reflection Groups acting on R^N, N = 2^k
# C.Hillar, 2020, Awecom Inc

import numpy as np

TEST = False

class Group():
    """ General group of size n acting on vectors of length N
    """
    def __init__(self, n=1, N=1):
        self.N = N
        self.n = n

    def act(self, g, x):  # trivial action
        return x

class Element():
    """ General element in a group
    """
    def __init__(self, group, normal_form=None):
        self.group = group
        if normal_form is not None:
            self.normal_form = normal_form
        else:
            self.normal_form = []
    
    def __call__(self, x):
        return self.group.act(self, x)        

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

    
###########################
# basic tests
###########################
if TEST:
    R = Reflection(2)
    S = Swap(2)
    x = np.array([0, 1, 2, 3])

    normal_form = [[0], [0, 0]]
    print("this is normal form %s for indentity element k = 2 acting on x = %s" % (normal_form, x))
    g = Element(R, normal_form=normal_form)
    y = R.act(g, x)
    print(y)

    normal_form = [[1], [0, 0]]
    print("this is normal form %s for reflecting total" % normal_form)
    g = Element(R, normal_form=normal_form)
    y = R.act(g, x)
    print(y)

    normal_form = [[1], [1, 1]]
    print("this is normal form %s for reflecting total then each of left and right halves" % normal_form)
    g = Element(R, normal_form=normal_form)
    y = R.act(g, x)
    print(y)

    normal_form = [[1], [0, 0]]
    print("this is normal form %s for swapping total" % normal_form)
    g = Element(S, normal_form=normal_form)
    y = S.act(g, x)
    print(y)

    ###################
    R = Reflection(3)
    S = Swap(3)
    x = np.array(range(8))

    print("reflecting total")
    normal_form = [[1], [0, 0], [0, 0, 0, 0]]
    g = Element(R, normal_form=normal_form)
    y = R.act(g, x)
    print(y)

    print("swapping all levels")
    normal_form = [[1], [1, 1], [1, 1, 1, 1]]
    g = Element(S, normal_form=normal_form)
    y = S.act(g, x)
    print(y)

    normal_form = [[0], [1, 1], [1, 0, 1, 1]]
    g = Element(S, normal_form=normal_form)
    print("this is normal form %s for swap group acting on x = %s" % (g.normal_form, x))
    y = S.act(g, x)
    print(y)

    g = S.rand_element()
    print("this is random normal form %s for swap group acting on x = %s" % (g.normal_form, x))
    y = S.act(g, x)
    print(y)

    print("this is random swap group action on x = %s" % x)
    print(S.rand_element(x=x))

    # make a data set 
    print("Data set as the orbit of some random reflection group actions of x = %s" % x)
    FR = DataFactory(R)
    x = np.array(range(8))
    data = FR.random_actions(x, 5)
    print(data)