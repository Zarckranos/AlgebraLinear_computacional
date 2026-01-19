import numpy as np

###############################################################################
# CONSTANTES                                                                  #
###############################################################################
AA = np.array([
    [ 2, -1,  1],
    [ 3,  3,  9],
    [ 3,  3,  5]
]) # Solução: [2.22, 1.19, -1.25]
BAA = np.array([2, -1, 4])

AB = np.array([
    [1, 2, 2, 2],
    [2, 4, 6, 8],
    [3, 6, 8, 10]
])
BAB = np.array([0, 0, 0, 0])


###############################################################################
# FUNÇÕES                                                                     #
###############################################################################
def print_system(A, b, precisao=2):
    n = len(b)
 
    print(f"\nSistema {50 * '='}")
    for i in range(n):
        linha_A = " ".join(f"{A[i, j]:{precisao+5}.{precisao}f}" for j in range(n))
        linha_b = f"{b[i]:{precisao+5}.{precisao}f}"

        print(f"| {linha_A} | * |  x{i+1}  | = | {linha_b} |")
    print(f"{58 * '='}\n")

def gen_system(n, m=None, low=-10, high=10):
    m = n if m is None else m

    A = np.random.randint(low, high + 1, size=(n, m))
    b = np.random.randint(low, high + 1, size=n)
    return A, b