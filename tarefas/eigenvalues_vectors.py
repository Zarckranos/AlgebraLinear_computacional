import numpy as np

###############################################################################
# TAREFA 08: Métodos de Potência                                              #
###############################################################################
def pot_regular(A, x, eps=1e-8):
    """
    Power Method
    
    :param A: Description
    :param x: Description
    :param eps: Description
    """
    x = x.astype(float)
    x = x / np.linalg.norm(x)

    lambda_old = 0

    for _ in range(1000):
        y = A @ x
        x_new = y / np.linalg.norm(y)

        # quociente de Rayleigh
        lambda_new = x_new.T @ A @ x_new

        if abs(lambda_new - lambda_old) < eps:
            break

        x = x_new
        lambda_old = lambda_new

    return lambda_new, x_new

def pot_inverse(A, x, eps=1e-8):
    """
    Inverse Power Method
    
    :param A: Description
    :param x: Description
    :param eps: Description
    """
    n = A.shape[0]
    x = x.astype(float)
    x = x / np.linalg.norm(x)

    lambda_old = 0

    for _ in range(1000):
        # resolve A y = x
        y = np.linalg.solve(A, x)
        x_new = y / np.linalg.norm(y)

        lambda_new = x_new.T @ A @ x_new

        if abs(lambda_new - lambda_old) < eps:
            break

        x = x_new
        lambda_old = lambda_new

    return lambda_new, x_new

def pot_desloc(A, x, mu, eps=1e-8):
    """
    Shifted Power Method
    
    :param A: Description
    :param x: Description
    :param eps: Description
    :param mu: Description
    """
    n = A.shape[0]
    x = x.astype(float)
    x = x / np.linalg.norm(x)

    lambda_old = 0
    I = np.eye(n)

    for _ in range(1000):
        # (A - mu I) y = x
        y = np.linalg.solve(A - mu * I, x)
        x_new = y / np.linalg.norm(y)

        lambda_new = x_new.T @ A @ x_new

        if abs(lambda_new - lambda_old) < eps:
            break

        x = x_new
        lambda_old = lambda_new

    return lambda_new, x_new

###############################################################################
# TAREFA 09: Método de Householder (Transformação de similaridade)            #
###############################################################################
def householder_reflection(x):
    """
    Cria o vetor de Householder v tal que: Hx = ||x|| e1
    """
    e1 = np.zeros_like(x)
    e1[0] = 1.0

    alpha = np.linalg.norm(x)
    if x[0] >= 0:
        alpha = -alpha

    v = x - alpha * e1
    v = v / np.linalg.norm(v)

    return v

def householder(A):
    """
    Implementar o método de Householder e faça o que se pede:
    1) imprima a matriz original (teste matrizes nxn com n > 6, simétricas e não
      simétricas);
    2) em cada passo,  imprima a matriz de householder, a matriz modificada pela
      transformação de similaridade, e a matriz de householder acumulada;
    3) no final do loop de execução  imprima 
        -  a matriz de saída (Tridiagonal se a matriz de entrada for simétrica,
           ou Upper-Hessemberg se a matriz de entrada não for simétrica);
        -  a matriz de Householder acumulada (final).
    """
    A = A.astype(float)
    n = A.shape[0]

    H_acc = np.eye(n)

    for k in range(n - 2):
        print(f"\nPASSO {k + 1} {40 * '='}")
        # Vetor a ser zerado abaixo da diagonal
        x = A[k + 1:, k]

        if np.linalg.norm(x[1:]) < 1e-12:
            continue

        v = householder_reflection(x)

        # Construção da matriz H_k
        Hk = np.eye(n)
        Hk[k + 1:, k + 1:] -= 2.0 * np.outer(v, v)

        print("Matriz de Householder H_k:\n", Hk)

        # Similaridade
        A = Hk @ A @ Hk.T

        print("Matriz após transformação A_k = H A H^T:\n", A)

        # Acumula transformações
        H_acc = H_acc @ Hk

        print("Matriz de Householder acumulada:\n", H_acc)

    return A, H_acc

###############################################################################
# TAREFA 10: Decomposição QR                                                  #
###############################################################################
def qr_decomposition(A):
    """
    Implementar a decomposição QR de uma matriz nxn.
    1) Entrar com uma matriz qualquer, A nxn
    2) Encontrar as matrizes Q (ortogonal) e R (triangular superior)  tal que A = Q R
    3) imprimir as matrizes Q e R e mostrar que o produto QR = A.
    """
    A = A.astype(float)
    n = A.shape[0]

    Q = np.zeros_like(A)
    R = np.zeros((n, n))

    for k in range(n):
        v = A[:, k].copy()

        for j in range(k):
            R[j, k] = np.dot(Q[:, j], A[:, k])
            v -= R[j, k] * Q[:, j]

        R[k, k] = np.linalg.norm(v)

        if R[k, k] == 0:
            raise ValueError("As colunas da matriz não são linearmente independentes.")

        Q[:, k] = v / R[k, k]

    return Q, R

###############################################################################
# TAREFA 11: Método QR para autovalores e autovetores                         #
###############################################################################
def qr_method(A, tol=1e-10, max_iter=1000):
    """
    Implemente o método QR para achar os autovalores e autovetores de matrizes.
    I. Matrizes simétricas
     - Entre com uma matriz simétrica A nxn e uma tolerância
     - Aplique o método QR diretamente sobre a matriz de entrada e encontre a
       matriz final (Diagonal) contendo os autovalores e a matriz acumulada 
       Q = Q1 . Q2 . Q3 . Q4 ... cujas colunas são os autovetores da matriz original.
     - Aplique o método de Householder primeiro (primeira caixa preta) e depois o
       método QR sobre a saída do Método de Householder, obtendo a matriz diagonal
       que contém os autovalores da matriz original. Obtenhas os autovetores da 
       matriz original como colunas da matriz P = H Q, onde H é a matriz acumulada
       de Householder e Q é a matriz acumulada no método QR.

    II. Matrizes não simétricas
     - Repita o que foi feito para matrizes simétricas. Porém, a matriz final do
       método QR é uma matriz Triangular Superior e Blocos (BUT).
     - Encontre os autovalores da matriz original, achando os autovalores dos
       blocos 2x2 ao longo da diagonal da BUT.
     - Ache os autovetores da matriz BUT e, em seguida, os autovetores da matriz
       original.
    """
    A_k = A.astype(float)
    n = A.shape[0]
    Q_acc = np.eye(n)

    for k in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_next = R @ Q
        Q_acc = Q_acc @ Q

        if np.linalg.norm(A_next - A_k) < tol:
            break

        A_k = A_next

    return A_k, Q_acc

def qr_symmetric(A, tol=1e-10):
    T, H = householder(A)
    D, Q = qr_method(T, tol)

    autovalores = np.diag(D)
    autovetores = H @ Q

    return autovalores, autovetores

def extract_eigenvalues_BUT(B, tol=1e-10):
    n = B.shape[0]
    eigenvalues = []

    i = 0
    while i < n:
        ...

    return np.array(eigenvalues)

def qr_nonsymmetric(A, tol=1e-10):
    H, P = householder(A)
    B, Q = qr_method(H, tol)

    autovalores = extract_eigenvalues_BUT(B)
    autovetores = P @ Q

    return autovalores, autovetores, B

###############################################################################
# TAREFA 12: Decomposição SVD                                                 #
###############################################################################
def svd():
    ...
