import numpy as np
from tarefas.system_resolutions import gauss_jordan
from utils.matrizes import *

def main(n=3):
    A, b = gen_system(int(n))
    print_system(A, b)

    b_, A_ = gauss_jordan(A, b)

    print("Solução: ")
    print(b_)
    print("A: \n", A_)

if __name__ == "__main__":
    main()