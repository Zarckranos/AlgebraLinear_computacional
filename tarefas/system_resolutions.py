import numpy as np
# TODO: preciso revisitar projeção de matrizes
###############################################################################
# TAREFA 01: Eliminação de Gauss                                              #
###############################################################################
def gauss_elimination(A, b, parcial_pivot=True):
    """
    Método de eliminação de Gauss com pivotação parcial (depois estender para
    pivotação total) para resolução de sistemas de equações algébricas lineares
    (Ax = b).

    Obs: Depois de testar com casos pequenos, teste com matrizes 10 x 10.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    col_perm = np.arange(n)
    for k in range(n - 1):
        if parcial_pivot:
            # Pivotação parcial
            max_index = np.argmax(np.abs(A[k:, k])) + k
            if A[max_index, k] == 0:
                print("Matriz singular!")
                return 0
            
            if max_index != k:
                A[[k, max_index]] = A[[max_index, k]]
                b[[k, max_index]] = b[[max_index, k]]
        else:
            # Pivotação total
            sub_matrix = np.abs(A[k:, k:])
            i_max, j_max = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)
            i_max += k
            j_max += k

            if abs(A[i_max, j_max]) < 1e-12:
                print("Matriz singular!")
                return 0
            
            # Troca de linhas
            if i_max != k:
                A[[k, i_max]] = A[[i_max, k]]
                b[[k, i_max]] = b[[i_max, k]]
            
            # Troca de colunas
            if j_max != k:
                A[:, [k, j_max]] = A[:, [j_max, k]]
                col_perm[[k< j_max]] = col_perm[[j_max, k]]
            
        # Eliminação
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Back-Substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    # Reorganiza solução no caso de pivotação total
    if not parcial_pivot:
        x_final = np.zeros(n)
        for i in range(n):
            x_final[col_perm[i]] = x[i]
        return x_final

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
def cholesky(A):
    """
    Implementar a decomposição SS^T de Cholesky de uma matriz simétrica e
    positiva definida.

    !Nota: Se, durante o processo de decomposição, no cálculo de algum elemento
    da diagonal envolver a raíz quadrada de um número negativo, o código deve
    escrever a mensagem "A Matriz não é positiva definida." e parar a execução.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Verificar simetria
    if not np.allclose(A, A.T, atol=1e-12):
        print("A matriz não é simétrica.")
        return None
    
    S = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            soma = sum(S[i, k] * S[j, k] for k in range(j))

            if i == j:
                valor = A[i, i] - soma
                if valor <= 0:
                    print("A Matriz não é positiva definida.")
                    return None
                S[i, j] = np.sqrt(valor)
            else:
                S[i, j] = (A[i, j] - soma) / S[j, j]
        
    return S

###############################################################################
# TAREFA 06: Ortogonalização de Gram-Schmidt                                  #
###############################################################################
def gram_schmidt(V, m):
    """
    Implemente o método de ortogonalização de Gram-Schmidt, faça o que se pede:
     - 1) Dado um conjunto de n vetores do Rm com n < m, estenda esse conjunto
       de vetores para achar uma base do Rm
     - 2) Use o processo de ortogonalização de Gram-Schmidt sobre a base
       estendida, para encontrar uma base ortonormal do Rm.
    """
    # Estende um conjunto LI para uma base de R^m
    basis = [v.astype(float) for v in V]

    for e in np.eye(m):
        candidate = basis + [e]
        M = np.column_stack(candidate)

        if np.linalg.matrix_rank(M) > len(basis):
            basis.append(e)

        if len(basis) == m:
            break
    
    print(f"Base estendida: {basis}")

    # Aplica o processo de Gram-Schmidt
    U = []
    for v in basis:
        u = v.astype(float).copy()
        for q in U:
            # Projeção: u = u - <u, q> * q
            u -= np.dot(q, u) * q

        if np.linalg.norm(u) > 1e-10:
            U.append(u / np.linalg.norm(u))

    return np.array(U) # base ortonormal

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