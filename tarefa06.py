import numpy as np
from tarefas.system_resolutions import gram_schmidt
from utils.matrizes import *

def main():
    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 1])

    V = np.array([v1, v2])
    m = 3

    print(f"V: {V}\nm = {m}")

    Q = gram_schmidt(V, m)
    print(f"Base ortonormal: \n{Q}")

    print("Verificação (Q @ Q^T) = I")
    print(Q @ Q.T)

if __name__ == "__main__":
    main()
