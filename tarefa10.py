import numpy as np
from tarefas.eigenvalues_vectors import qr_decomposition
from utils.matrizes import *

def main(n=3):
    A, _ = gen_system(n)
    print("\nMatriz original A:\n", A)

    Q, R = qr_decomposition(A)

    print("\nMatriz Q (ortogonal):")
    print(Q)

    print("\nMatriz R (triangular superior):")
    print(R)

    print("\nProduto QR:")
    print(Q @ R)

    print("\nVerificação Q^TQ:")
    print(Q.T @ Q)