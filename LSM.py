import numpy as np
import matplotlib.pyplot as plt

# Экспериментальные данные (пример)
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([2.3, 4.1, 5.8, 8.5, 11.2, 13.9], dtype=float)

# Количество точек
n = len(x)

# --- 1. Линейная аппроксимация y = a*x + b ---
A_lin = np.vstack([x, np.ones(n)]).T
coeff_lin, residuals, _, _ = np.linalg.lstsq(A_lin, y, rcond=None)
a_lin, b_lin = coeff_lin

def f_lin(x): 
    return a_lin * x + b_lin

# --- 2. Степенная аппроксимация y = beta * x^a ---
# Линеаризация: ln(y) = ln(beta) + a*ln(x)
ln_x = np.log(x)
ln_y = np.log(y)
A_pow = np.vstack([ln_x, np.ones(n)]).T
coeff_pow, _, _, _ = np.linalg.lstsq(A_pow, ln_y, rcond=None)
a_pow, ln_beta = coeff_pow
beta_pow = np.exp(ln_beta)

def f_pow(x): 
    return beta_pow * x**a_pow

# --- 3. Показательная аппроксимация y = beta * exp(a*x) ---
# Линеаризация: ln(y) = ln(beta) + a*x
A_exp = np.vstack([x, np.ones(n)]).T
coeff_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_y, rcond=None)
a_exp, ln_beta_exp = coeff_exp
beta_exp = np.exp(ln_beta_exp)

def f_exp(x): 
    return beta_exp * np.exp(a_exp * x)

# --- 4. Квадратичная аппроксимация y = a*x^2 + b*x + c ---
A_quad = np.vstack([x**2, x, np.ones(n)]).T
coeff_quad, _, _, _ = np.linalg.lstsq(A_quad, y, rcond=None)
a_quad, b_quad, c_quad = coeff_quad

def f_quad(x): 
    return a_quad * x**2 + b_quad * x + c_quad

# --- Вычисление суммы квадратов отклонений ---
def sum_squared_errors(f):
    return np.sum((f(x) - y)**2)

S_lin = sum_squared_errors(f_lin)
S_pow = sum_squared_errors(f_pow)
S_exp = sum_squared_errors(f_exp)
S_quad = sum_squared_errors(f_quad)

# --- Вывод результатов ---
print("Коэффициенты и сумма квадратов отклонений:")
print(f"Линейная: y = {a_lin:.4f}x + {b_lin:.4f}, S = {S_lin:.4f}")
print(f"Степенная: y = {beta_pow:.4f} * x^{a_pow:.4f}, S = {S_pow:.4f}")
print(f"Показательная: y = {beta_exp:.4f} * exp({a_exp:.4f} * x), S = {S_exp:.4f}")
print(f"Квадратичная: y = {a_quad:.4f}x^2 + {b_quad:.4f}x + {c_quad:.4f}, S = {S_quad:.4f}")

# --- Построение графиков ---

x_plot = np.linspace(min(x)*0.9, max(x)*1.1, 200)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Экспериментальные данные', zorder=5)
plt.plot(x_plot, f_lin(x_plot), label='Линейная')
plt.plot(x_plot, f_pow(x_plot), label='Степенная')
plt.plot(x_plot, f_exp(x_plot), label='Показательная')
plt.plot(x_plot, f_quad(x_plot), label='Квадратичная')
plt.title('Аппроксимация экспериментальных данных')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Отдельное окно с графиками всех функций вместе (уже сделано выше)
