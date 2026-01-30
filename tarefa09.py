import numpy as np
from tarefas.eigenvalues_vectors import householder
from utils.matrizes import *

def main(n=3, m=3, simetric=False):
    # np.random.seed(42)
    A, _ = gen_system(n, m)
    if simetric:
        A = (A + A.T) / 2

    print("\nMatriz original A:\n", A)
    A_k, Q = householder(A)

    print(f"\nRESULTADO FINAL {40 * '='}\n")
    print("Matriz tridiagonal:\n", A_k)
    print("\nMatriz de Householder acumulada:\n", Q)

if __name__ == "__main__":
    main()