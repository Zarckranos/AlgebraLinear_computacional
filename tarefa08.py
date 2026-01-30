import numpy as np
from tarefas.eigenvalues_vectors import pot_regular, pot_inverse, pot_desloc
from utils.matrizes import *

def main(n=3):
    # A, _ = gen_system(n)
    A = np.array([
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2]
    ], dtype=float)
    print("\nA:\n", A)

    x0 = np.array([1, 1, 1], dtype=float)

    # Potência regular
    lam1, v1 = pot_regular(A, x0)
    print("Potência regular:")
    print("Autovalor:", lam1)
    print("Autovetor:", v1)

    # Potência inversa
    lam2, v2 = pot_inverse(A, x0)
    print("\nPotência inversa:")
    print("Autovalor:", lam2)
    print("Autovetor:", v2)

    # Potência com deslocamento
    mu = 2.5
    lam3, v3 = pot_desloc(A, x0, mu)
    print("\nPotência com deslocamento:")
    print("Autovalor:", lam3)
    print("Autovetor:", v3)

if __name__ == "__main__":
    main()