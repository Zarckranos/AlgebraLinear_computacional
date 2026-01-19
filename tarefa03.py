import numpy as np
from tarefas.system_resolutions import lu_decomposition
from utils.matrizes import *

def main(n=3):
    A, b = gen_system(int(n))
    print_system(A, b)

    x = lu_decomposition(A, b)

    print("Solução do sistema:")
    print(x)

if __name__ == "__main__":
    main()