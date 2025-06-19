import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Q = 15000
U = 200
A = 10
T1 = 8.5

def f(T2):
    if T2 <= 0 or T2 == T1:
        return np.nan
    return U * A * (T1 - T2) / np.log(T1 / T2) - Q

def df(T2, h=1e-5):
    return (f(T2 + h) - f(T2 - h)) / (2 * h)

def bisseccao(a, b, tol=1e-6, max_iter=100):
    print("Método da Bissecção:")
    print("Iter |     a     |     b     |   c (médio)   |  f(c)")

    iteracoes = []
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        erro = abs(fc)
        iteracoes.append([i, a, b, c, fc, erro])
        print(f"{i:4d} | {a:.6f} | {b:.6f} | {c:.6f} | {fc:.6f}")
        if erro < tol or (b - a) / 2 < tol:
            return c, iteracoes
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, iteracoes

def newton_raphson(x0, tol=1e-6, max_iter=100):
    print("\nMétodo de Newton-Raphson:")
    print("Iter |     x     |    f(x)    |   Erro")

    iteracoes = []
    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            print("Derivada zero. Método falhou.")
            return None, iteracoes
        x1 = x0 - fx / dfx
        erro = abs(x1 - x0)
        iteracoes.append([i, x0, fx, erro])
        print(f"{i:4d} | {x0:.6f} | {fx:.6f} | {erro:.6e}")
        if erro < tol:
            return x1, iteracoes
        x0 = x1
    return x0, iteracoes

T2_vals = np.linspace(0.1, 8.4, 500)
f_vals = [f(T2) for T2 in T2_vals]

plt.figure(figsize=(8, 5))
plt.plot(T2_vals, f_vals, label='f(T2)')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Análise Gráfica da Função f(T2)")
plt.xlabel("T2")
plt.ylabel("f(T2)")
plt.grid(True)
plt.legend()
plt.savefig("grafico_f_T2.png", dpi=300)
plt.show()

def plot_erro(iter_b, iter_n):
    plt.figure(figsize=(10, 5))

    erros_b = [abs(linha[4]) for linha in iter_b]
    erros_n = [linha[3] for linha in iter_n]

    plt.semilogy(erros_b, label="Bissecção", marker='o')
    plt.semilogy(erros_n, label="Newton-Raphson", marker='x')
    plt.xlabel("Iteração")
    plt.ylabel("Erro (escala log)")
    plt.title("Convergência dos Métodos")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergencia_erros.png", dpi=300)
    plt.show()

raiz_bisseccao, it_bisseccao = bisseccao(4.0, 8.0)
raiz_newton, it_newton = newton_raphson(6.0)

df_bisseccao = pd.DataFrame(it_bisseccao, columns=["Iter", "a", "b", "c", "f(c)", "Erro"])
df_newton = pd.DataFrame(it_newton, columns=["Iter", "x", "f(x)", "Erro"])

df_bisseccao.to_csv("tabela_bisseccao.csv", index=False)
df_newton.to_csv("tabela_newton.csv", index=False)

plot_erro(it_bisseccao, it_newton)

print(f"\nRaiz encontrada (Bissecção): {raiz_bisseccao:.6f}")
print(f"Raiz encontrada (Newton-Raphson): {raiz_newton:.6f}")
