# !--- DANYLO ZHERZDIEV 196765 ---! #
# !---      Projekt nr. 2      ---! #

import numpy as np
import matplotlib.pyplot as plt
import time

# Uzupelnienie macierzy
def matrix_filling(N, a1,a2,a3):
    A = np.zeros([N,N])
    for i in range (N):
        for j in range (N):
            if i == j:
                A[i][j] = a1
            if ((i == (j - 1)) | (i - 1 == j) ):
                A[i][j] = a2
            if ((i == (j - 2)) | (i - 2 == j) ):
                A[i][j] = a3
    return A

# Metoda iteracyjna Jacobiego z residuum
def jacobi_method(A, b, tol=1e-9, max_iter=50):
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    residuum = []
    start_time = time.time()
    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        res = np.linalg.norm(b - np.dot(A, x_new))
        residuum.append(res)
        if res < tol:
            end_time = time.time() - start_time
            return x_new, k + 1, residuum, end_time
        x = x_new
    end_time = time.time() - start_time
    return x, max_iter, residuum, end_time

# Metoda Gaussa-Seidla z residuum
def gauss_seidel(A, b, tol=1e-9, max_iter=50):
    x = np.zeros_like(b)
    n = len(b)
    residuum = []
    start_time = time.time()
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        res = np.linalg.norm(b - np.dot(A, x_new))
        residuum.append(res)
        if res < tol:
            end_time = time.time() - start_time
            return x_new, k + 1, residuum, end_time
        x = x_new
    end_time = time.time() - start_time
    return x, max_iter, residuum, end_time

# Funkcja rozkładu LU
def lu_decomposition(A):
    N = len(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    for i in range(N):
        L[i, i] = 1  # Wstawienie jedynki na diagonali L
        for j in range(i, N):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])  # Obliczenie elementów U
        for j in range(i + 1, N):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]  # Obliczenie elementów L
    
    return L, U

# Funkcja rozwiązywania układu równań z rozkładu LU
def lu_solve(L, U, b):
    # Rozwiązywanie układu Ly = b
    N = len(b)
    y = np.zeros_like(b)
    for i in range(N):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Rozwiązywanie układu Ux = y
    x = np.zeros_like(b)
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x

# Rozwiązywanie układu równań
def calculate_LU(A, b):
    start_time = time.time()
    L, U = lu_decomposition(A)
    x_lu = lu_solve(L, U, b)
    end_time = time.time()

    return start_time, end_time, x_lu

# Funkcja obliczania residuum
def calculate_residuum(A, x, b):
    return np.linalg.norm(b - np.dot(A, x))





# Zadanie A-B
index = 196765

f = (((index // 10) // 10 )// 10) % 10  # trzecia cyfra
e = index % 10  # ostatnia cyfra
c = (index // 10) % 10  # przedostatnia
a1 = 5 + e
a2 = a3 = -1

N = 1200 + 10*c + e

b = np.array([np.sin(n * (f + 1)) for n in range(1, N + 1)])

A1 = matrix_filling(N, a1, a2, a3)
result_jacobi1 = jacobi_method(A1, b)
result_gauss_seidel1 = gauss_seidel(A1, b)


x_jacobi, iter_jacobi, res_jacobi, time_jacobi = result_jacobi1
x_gauss_seidel, iter_gauss_seidel, res_gauss_seidel, time_gauss_seidel = result_gauss_seidel1

print("Macierz A: \n", A1)
print("\nb: \n", b)





# Zadanie C
a1 = 3
A2 = matrix_filling(N, a1, a2, a3)
print("Macierz A: \n", A2)
result_jacobi2 = jacobi_method(A2, b)
result_gauss_seidel2 = gauss_seidel(A2, b)

x_jacobi2, iter_jacobi2, res_jacobi2, time_jacobi2 = result_jacobi2
x_gauss_seidel2, iter_gauss_seidel2, res_gauss_seidel2, time_gauss_seidel2 = result_gauss_seidel2

# Wynik dla (a1=12, a2=-1, a3=-1)
print(f"Metoda Jacobiego (a1=12): {iter_jacobi} iteracji, czas: {time_jacobi:.4f} s")
print(f"Final residual norm: {res_jacobi[-1]:.4e}")
print(f"Metoda Gaussa-Seidla (a1=12): {iter_gauss_seidel} iteracji, czas: {time_gauss_seidel:.4f} s")
print(f"Final residual norm: {res_gauss_seidel[-1]:.4e}\n")

# Wynik dla (a1=3, a2=-1, a3=-1)
print(f"Metoda Jacobiego (a1=3): {iter_jacobi2} iteracji, czas: {time_jacobi2:.4f} s")
print(f"Final residual norm: {res_jacobi2[-1]:.4e}")
print(f"Metoda Gaussa-Seidla (a1=3): {iter_gauss_seidel2} iteracji, czas: {time_gauss_seidel2:.4f} s")
print(f"Final residual norm: {res_gauss_seidel2[-1]:.4e}\n")





# Zadanie D
# Norma residuum
start_time, end_time, x_lu = calculate_LU(A2, b)
residuum_lu = calculate_residuum(A2, x_lu, b)

# Wyniki
print(f"Norma residuum (LU): {residuum_lu:.4e}")
print(f"Czas wykonania (LU): {end_time - start_time:.4f} s\n")





# Zadanie E 
# Mierzony czas dla różnych N
N_values = [100, 500, 1000, 2000, 3000]
times_jacobi = []
times_gauss_seidel = []
times_lu = []
sum_times_jacobi = 0
sum_times_gauss_seidel = 0
sum_times_lu = 0

for N in N_values:
    b = np.array([np.sin(n * 1) for n in range(1, N + 1)])
    A = matrix_filling(N, 10, -1, -1)
    
    print("\nWynik dla          N = 100                 N = 500                 N = 1000                    N = 2000                    N = 3000\n")
    # Jacobi
    _, _, _, time_jacobi = jacobi_method(A, b)
    times_jacobi.append(time_jacobi)
    print("Jacobi TIME: ", times_jacobi )
    
    # Gauss-Seidel
    _, _, _, time_gauss_seidel = gauss_seidel(A, b)
    times_gauss_seidel.append(time_gauss_seidel)
    print("Gauss TIME: ", times_gauss_seidel )
    
    # LU
    start_time, end_time, _ = calculate_LU(A, b)
    times_lu.append(end_time - start_time)
    print("LU TIME: ", times_lu, "\n")

sum_times_jacobi = sum(times_jacobi)
sum_times_gauss_seidel = sum(times_gauss_seidel)
sum_times_lu = sum(times_lu)

print("\nCzas na wykonanie metoda Jacobi dla wszystkich N:", sum_times_jacobi)
print("Czas na wykonanie metoda Gauss-Seidel dla wszystkich N:", sum_times_gauss_seidel)
print("Czas na wykonanie LU dla wszystkich N:", sum_times_lu)









# Wykresy

# Funkcja do rysowania wykresu normy residuum
def plot_residuum(iter_jacobi, res_jacobi, iter_gauss_seidel, res_gauss_seidel, label_jacobi, label_gauss_seidel, title):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, iter_jacobi + 1), res_jacobi, label=label_jacobi, marker='o', markersize=3)
    plt.semilogy(range(1, iter_gauss_seidel + 1), res_gauss_seidel, label=label_gauss_seidel, marker='s', markersize=3)
    
    plt.xlabel("Liczba iteracji")
    plt.ylabel("Norma residuum")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Funkcja do rysowania wykresu czasu wykonania   
def plot_execution_time(N_values, times_jacobi, times_gauss_seidel, times_lu, title, log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, times_jacobi, label='Jacobi', marker='o')
    plt.plot(N_values, times_gauss_seidel, label='Gauss-Seidel', marker='s')
    plt.plot(N_values, times_lu, label='LU', marker='d')
    
    plt.xlabel("Liczba niewiadomych (N)")
    plt.ylabel("Czas wykonania (s)")
    plt.title(title)
    
    if log_scale:
        plt.yscale('log')
    
    plt.legend()
    plt.grid(True)
    plt.show()


# Wykres normy residuum dla (a1=12, a2=-1, a3=-1) i (a1=3, a2=-1, a3=-1)
plot_residuum(iter_jacobi, res_jacobi, iter_gauss_seidel, res_gauss_seidel, 
              'Jacobi (a1=10)', 'Gauss-Seidel (a1=10)', 
              "Porównanie metod iteracyjnych dla a1 = 10")

plot_residuum(iter_jacobi2, res_jacobi2, iter_gauss_seidel2, res_gauss_seidel2, 
              'Jacobi (a1=3)', 'Gauss-Seidel (a1=3)', 
              "Porównanie metod iteracyjnych dla a1 = 3")

# Wykres czasu wykonania metod liniowa skalę osi Y
plot_execution_time(N_values, times_jacobi, times_gauss_seidel, times_lu, 
                    "Czas wykonania metod iteracyjnych i LU w zależności od N")

# Wykres z logarytmiczną skalą Y
plot_execution_time(N_values, times_jacobi, times_gauss_seidel, times_lu, 
                    "Czas wykonania metod iteracyjnych i LU w zależności od N (skala logarytmiczna)", 
                    log_scale=True)

