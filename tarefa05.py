import numpy as np
from tarefas.system_resolutions import cholesky
from utils.matrizes import *

def main():
    A = np.array([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ])
    print(f"Matriz A:\n{A}")

    S = cholesky(A)

    if S is not None:
        print(f"Matriz S: \n{S}")
        print(f"\n(S @ S^T): \n{S @ S.T}")

if __name__ == "__main__":
    main()