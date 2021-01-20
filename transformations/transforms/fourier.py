import numpy as np

#TODO: Either convert these into proper datasets or delete

def gen_1D_fourier_variant(n_samples, 
                            dim, 
                            complex_input=False,
                            project_input=False,
                            project_output=False,
                            variant='fourier', 
                            standardize=False,
                            seed=0):
    
    np.random.seed(seed)
    
    X = np.random.uniform(-1, 1, size=(n_samples, dim))
    
    if complex_input:
        X += 1j * np.random.uniform(-1, 1, size=(n_samples, dim))
        
    F = np.fft.fft(X, axis=-1)
    
    if complex_input and project_input:
        X = np.hstack([X.real, X.imag])
    
    if variant == 'fourier':
        if project_output:
            F = np.hstack([F.real, F.imag])
        return X, F
    
    elif variant == 'power-spectrum':
        PS = np.abs(F) ** 2
        return X, PS
    
    elif variant == 'bispectrum':
        B = []
        for f in F:
            b = np.zeros(dim + 1, dtype=np.complex)
            b[0] = np.conj(f[0] * f[0]) * f[0]
            b[1:-1] = np.conj(f[:-1] * f[1]) * f[1:]
            b[dim] = np.conj(f[dim - 1] * f[1]) * f[0]
            B.append(b)
        B = np.array(B)
        if project_output:
            B = np.hstack([B.real, B.imag])
        return X, B
    
    else:
        raise ValueError('Invalid dataset type')


def bispectrum_2d(X, truncated=True, flatten=True):
    """
    Author: Chris Hillar
    Computes the 2D bispectrum for an M x N fourier transform
    X: numpy (T x M x N)
    OUTPUT: (T x M x N x M x N)
    if truncated is True then
    OUTPUT: (T x (m * n + 2)) numpy array
    F(i,j) = F*(m-i,n-j)
    """
    F = np.fft.fft2(X)
    m = F.shape[1]
    n = F.shape[2]
    T = F.shape[0]

    if truncated is True:  # compute smaller but sufficient num bispect coeffs
        B = np.zeros((T, m * n + 2), np.complex)
        B[:, 0] = np.conj(F[:, 0, 0] * F[:, 0, 0]) * F[:, 0, 0]
        B[:, 1:m] = np.conj(F[:, :-1, 0] * F[:, 1, 0][:, None]) * F[:, 1:, 0]
        B[:, m] = np.conj(F[:, m - 1, 0] * F[:, 1, 0]) * F[:, 0, 0]
        B[:, m + 1 : m + n] = np.conj(F[:, 0, :-1] * F[:, 0, 1][:, None]) * F[:, 0, 1:]
        B[:, m + n] = np.conj(F[:, 0, n - 1] * F[:, 0, 1]) * F[:, 0, 0]
        tmp = (
            np.conj(F[:, 0:1, 1:] * F[:, 1:, 0:1]).T * F[:, 1:, 1:].T
        )  # [2,1,0].reshape(T, (m - 1) * (n - 1))
        np.swapaxes(tmp, 0, 2)
        np.swapaxes(tmp, 1, 2)
        B[:, m + n + 1 :] = tmp.reshape(T, (m - 1) * (n - 1))
        return B

    B = np.zeros((T, m, n, m, n), dtype=np.complex128)
    for i in range(m):  # TODO: loop could be made faster with numpy tricks?
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    B[:, i, j, k, l] = (
                        F[:, i, j] * F[:, k, l] * F[:, (i + k) % m, (j + l) % n].conj()
                    )
    if flatten is True:
        return B.reshape(T, m * n * m * n)
    return B


def bispectrum_1d(f, n=None):
    """
    Author: Chris Hillar
    compute bispectrum of f: {0,1,...,n-1} = Z/nZ -> C

    n:        positive integer
    f:        complex vector of length n

    Computes a 1x(n+1) vector B = [B(0,0), B(0,1), B(1,1), ..., B(n-1,1)]
    in which B(k1,k2) = f^(k1)* f^(k2)* f^(k1+k2)

    Here f^ is the Fourier Transform of f and * means complex conjugation
    """

    assert f.ndim == 1, "input has to be one-dimensional!"
    if n is None:
        n = len(f)

    B = np.zeros(n + 1, dtype=np.complex)
    F = np.fft.fft(f, n)

    B[0] = np.conj(F[0] * F[0]) * F[0]
    B[1:-1] = np.conj(F[:-1] * F[1]) * F[1:]
    B[n] = np.conj(F[n - 1] * F[1]) * F[0]

    return B



def gen_random_translations_1d(
    n_primitives,
    n_translations,
    primitive_size,
    data_size,
    get_bispecs=False,
    concat_bispecs=False,
):
    data = np.zeros((n_translations * n_primitives, data_size))
    labels = np.zeros((n_translations * n_primitives, n_primitives))
    bispecs = []
    max_stride = data_size - primitive_size
    idx = 0
    for n in range(n_primitives):
        primitive = np.random.uniform(-1, 1, size=primitive_size)
        for i in range(n_translations):
            shift = np.random.randint(max_stride)
            translated = np.zeros(data_size)
            translated[shift : shift + primitive_size] = primitive
            data[idx] = translated
            labels[idx][n] = 1
            if get_bispecs:
                bispecs.append(bispectrum_1d(translated))
            idx += 1
    shuffle_idx = np.random.choice(
        range(data.shape[0]), replace=False, size=data.shape[0]
    )
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]
    if bispecs:
        bispecs = np.array(bispecs)
        if concat_bispecs:
            bispecs = np.hstack([bispecs.real, bispecs.imag])
        bispecs = bispecs[shuffle_idx]
        return data, labels, bispecs
    else:
        return data, labels
