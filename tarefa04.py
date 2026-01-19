from tarefas.system_resolutions import rref
from utils.matrizes import *

def main(n=3, m=3):
    A, _ = gen_system(n, m)
    print("\nA:\n", A)

    R, rank, pivots = rref(A)

    print("R: \n", R)
    print(f"Rank da matriz: {rank}")
    print(f"Pivots: {pivots}")

if __name__ == "__main__":
    main()