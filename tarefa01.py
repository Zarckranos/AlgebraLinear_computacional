import numpy as np
from tarefas.system_resolutions import gauss_elimination
from utils.matrizes import *

def main(n=10):
    A, b = gen_system(int(n))
    print_system(A, b)

    x = gauss_elimination(A, b)

    print("Solução do sistema:")
    print(x)

if __name__ == "__main__":
    main()
