import numpy as np
from tarefas.eigenvalues_vectors import qr_symmetric, qr_nonsymmetric
from utils.matrizes import *

def main(n=3, m=3):
    A, _ = gen_system(n, m)

    # Matriz simétrica
    A = np.array([
        [4, 1, 2],
        [1, 2, 0],
        [2, 0, 3]
    ])
    print("\nMatriz A:\n", A)

    vals, vecs = qr_symmetric(A)

    print("Autovalores (simétrica):")
    print(vals)

    print("\nAutovetores:")
    print(vecs)

    # Matriz não simétrica
    B = np.array([
        [1, 2, 3],
        [0, 4, 5],
        [0, -6, 7]
    ])

if __name__ == "__main__":
    main()