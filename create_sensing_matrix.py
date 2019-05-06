import numpy as np
from scipy.linalg import orth
from numpy.random import randn

W, H = 32, 32
n = W * H
cr = 80
m = n // cr


def main():
    sensing_matrix = randn(m, n)
    sensing_matrix = orth(sensing_matrix.transpose()).transpose()
    sensing_matrix = np.expand_dims(sensing_matrix, axis=0)
    sensing_matrix = np.repeat(sensing_matrix, 3, axis=0)
    np.save('./sensing_matrix_cr{}_w{}_h{}.npy'.format(cr, W, H), sensing_matrix)


if __name__ == '__main__':
    main()
