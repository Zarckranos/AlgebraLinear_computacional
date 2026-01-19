import os

def menu():
    print(f"\nÁlgebra Linear Computacional {30 * "="}")
    print(" 0 - SAIR")
    print("> Resolução de Sistemas de Equações Algébricas Lineares")
    print(" 1 - Eliminação de Gauss")
    print(" 2 - Gauss-Jordan")
    print(" 3 - Decomposição LU")
    print(" 4 - RREF")
    print(" 5 - Decomposição de Cholesky")
    print(" 6 - Ortogonalização de Gram-Schmidt")
    print(" 7 - Mínimos quadrados")
    print("> Autovalores e Autovetores")
    print(" 8 - Métodos de Potência")
    print(" 9 - Método de Householder (Transformação de similaridade)")
    print(" 10 - Decomposição QR")
    print(" 11 - Método QR para autovalores e autovetores")
    print(" 12 - Decomposição SVD")
    print("> Métodos Iterativos para Solução de Sistemas de Equações Algébricas Lineares")
    print(" 13 - SOR")
    print(" 14 - Gradientes conjugados")


if __name__ == "__main__":
    while True:
        menu()
        case = input("Escolha uma opção: ")
        os.system('cls')

        if case == "1":
            import tarefa01
            n = input("Selecione o tamanho da matriz (n x n): ")
            tarefa01.main(n)
        elif case == "2":
            import tarefa02
            n = input("Selecione o tamanho da matriz (n x n): ")
            tarefa02.main(n)
        elif case == "3":
            import tarefa03
            n = input("Selecione o tamanho da matriz (n x n): ")
            tarefa03.main(n)
        elif case == "4":
            import tarefa04
            try:
                n = int(input("Selecione o valor de n: "))
                m = int(input("Selecione o valor de m: "))
            except:
                print("VALOR DIGITADO INCORRETO, APENAS INTEIROS!")
            tarefa04.main(n, m)
        elif case == "5":
            import tarefa05
            tarefa05.main()
        elif case == "7":
            import tarefa07
            try:
                n = int(input("Selecione o valor de n: "))
                m = int(input("Selecione o valor de m: "))
            except:
                print("VALOR DIGITADO INCORRETO, APENAS INTEIROS!")
            tarefa07.main(n, m)
        elif case == "0":
            print("Encerrando...")
            break
        else:
            print("Opção inválida!")
