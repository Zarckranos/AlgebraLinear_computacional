from tarefas.system_resolutions import least_squares
from utils.matrizes import *

def main(n=3, m=3):
    A, b = gen_system(n, m)
    print(f"A:\n {A}")
    print(f"b:\n {b}")

    x = least_squares(A, b)

    print("Solução do sistema:")
    print(x)

if __name__ == "__main__":
    main()