import numpy as np
# TODO: preciso revisitar projeção de matrizes
###############################################################################
# TAREFA 01: Eliminação de Gauss                                              #
###############################################################################
def gauss_elimination(A, b):
    """
    Método de eliminação de Gauss com pivotação parcial (depois estender para
    pivotação total) para resolução de sistemas de equações algébricas lineares
    (Ax = b).

    Obs: Depois de testar com casos pequenos, teste com matrizes 10 x 10.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n - 1):
        # Pivotação parcial
        max_index = np.argmax(np.abs(A[k:, k])) + k
        if A[max_index, k] == 0:
            print("Matriz singular!")
            return 0
        
        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Back-Substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

###############################################################################
# TAREFA 02: Gauss-Jordan                                                     #
###############################################################################
def gauss_jordan(A, b):
    """
    Implementar o método de Gauss-Jordan para resolução de sistemas Ax = b.
    Obs. Aproveitar as estratégias de pivotação da Tarefa 01.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n):
        # Pivotação parcial
        max_index = np.argmax(np.abs(A[k:, k])) + k
        if A[max_index, k] == 0:
            print("Matriz singular!")
            return 0

        if max_index != k:
            A[[k, max_index]] = A[[max_index, k]]
            b[[k, max_index]] = b[[max_index, k]]

        # Normalização da linha do pivô
        pivot = A[k, k]
        A[k, :] /= pivot
        b[k] /= pivot

        # Eliminação em todas as outras linhas
        for i in range(n):
            if i != k:
                m = A[i, k]
                A[i, :] -= m * A[k, :]
                b[i] -= m * b[k]

    return b, A

###############################################################################
# TAREFA 03: Decomposição LU e resolução de sistemas lineares                 #
###############################################################################
def lu_decomposition(A, b):
    """
    Implementar a decomposição LU de uma matriz e utilizá-la na solução de um
    sistema linear.

    !Nota: às vezes a matriz original não tem decomposição LU ou tem uma decomposição
    PLU onde P é uma matriz de permutação. Vamos considerar apenas os casos em
    que a matriz tem decomposição LU sem recorrer à permutação de linhas.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        for j in range(k, n):  # linha k de U
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        
        if U[k, k] == 0:
            print("Matriz não admite dedcomposição sem permutação!")
            return 0

        for i in range(k + 1, n):  # coluna k de L
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    
    print("L =\n", L)
    print("U =\n", U)
    print("LU =\n", L @ U)

    # Substituição (Ly = b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Back-Substitution (Ux = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

###############################################################################
# TAREFA 04: Reduced Row Echelon Form                                         #
###############################################################################
def rref(A):
    """
    Método para determinar o RREF de uma matriz mxn e, testando em matrizes com
    m > n, m < n e m=n, determine:
     - 1) RREF da matriz
     - 2) O posto da matriz (rank)
     - 3) A dimensão do espaço nulo da matriz
    """
    A = A.astype(float).copy()
    n, m = A.shape

    row = 0
    pivots = []
    tol = 1e-10
    for j in range(m):
        pivot_row = np.argmax(np.abs(A[row:, j])) + row
        if abs(A[pivot_row, j]) < tol:
            continue

        A[[row, pivot_row]] = A[[pivot_row, row]]
        A[row] /= A[row, j]

        for i in range(n):
            if i != row:
                A[i] -= A[i, j] * A[row]

        pivots.append(j)
        row += 1

        if row == n:
            break
        
    A[np.abs(A) < tol] = 0.0
    rank = len(pivots)
    print(f"Dimensão do espaço nulo: {m - rank}")
    return A, rank, pivots

###############################################################################
# TAREFA 05: Decomposição de Cholesky                                         #
###############################################################################
def cholesky():
    """
    Implementar a decomposição SS^T de Cholesky de uma matriz simétrica e
    positiva definida.

    !Nota: Se, durante o processo de decomposição, no cálculo de algum elemento
    da diagonal envolver a raíz quadrada de um número negativo, o código deve
    escrever a mensagem "A Matriz não é positiva definida." e parar a execução.
    """
    ...

###############################################################################
# TAREFA 06: Ortogonalização de Gram-Schmidt                                  #
###############################################################################
def gram_schmidt():
    """
    Implemente o método de ortogonalização de Gram-Schmidt, faça o que se pede:
     - 1) Dado um conjunto de n vetores do Rm  com n < m, estenda esse conjunto
       de vetores para achar uma base do Rm
     - 2) Use o processo de ortogonalização de Gram-Schmidt sobre a base
       estendida, para encontrar uma base ortonormal do Rm.
    """
    ...

###############################################################################
# TAREFA 07: Mínimos quadrados                                                #
###############################################################################
def least_squares(A, b): 
    """
    Considere uma matriz A mxn com m > n  e um vetor b pertencente ao Rm.
    Encontre uma solução aproximada x*  para o problema Ax = b que minimize a
    norma quadrada do vetor Ax* - b.
    Aplique este tipo de solução ao problema de regressão linear.
    """
    A = A.astype(float)
    b = b.astype(float)

    ATA = A.T @ A
    ATb = A.T @ b

    x = np.linalg.solve(ATA, ATb)
    return x