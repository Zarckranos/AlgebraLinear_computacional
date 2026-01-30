import numpy as np
from tarefas.eigenvalues_vectors import svd
from utils.matrizes import *

def main(n=3):
    # A = np.array([
    #     [1, 2],
    #     [3, 4],
    #     [5, 6]],
    # dtype=float)
    A = np.array([
        [1,  2,  3],
        [4,  5,  6],
        [7,  8, 10],
        [2,  3,  4]
    ], dtype=float)

    U, Sigma, V = svd(A)

    print("\n Verificação do SVD ============")
    print("\nMatriz A:\n", A)
    print("U =\n", U)
    print("\nSigma =\n", Sigma)
    print("\nV =\n", V)

    A_recon = U @ Sigma @ V.T
    print("\nU @ Sigma @ V^T =\n", A_recon)

    print("\n Comparação com numpy ============")
    U1, S1, V1 = np.linalg.svd(A, full_matrices=True)

    print("Erro reconstrução:")
    print(np.linalg.norm(A - U @ Sigma @ V.T))

    print("Valores singulares (tarefa12):", np.diag(Sigma))
    print("Valores singulares (numpy):", S1)

if __name__ == "__main__":
    main()