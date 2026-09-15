# ============================================================
# ЗАДАНИЕ 1. Энтропия Шеннона
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Функция вычисления энтропии (НЕ МЕНЯЙТЕ) ---
def entropy_binary(p):
    """Энтропия бинарной случайной величины H(p) в битах."""
    p = np.asarray(p, dtype=float)
    H = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    H[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    return H

# --- Шаг 1. График энтропии (НЕ МЕНЯЙТЕ) ---
p_values = np.linspace(0.01, 0.99, 200)
H_values = entropy_binary(p_values)

plt.figure(figsize=(8, 5))
plt.plot(p_values, H_values, 'b-', linewidth=2, label='H(p)')
plt.axvline(0.5, color='r', linestyle='--', alpha=0.7, label='Максимум при p=0.5')
plt.axhline(1.0, color='g', linestyle=':', alpha=0.5, label='H_max = 1 бит')
plt.xlabel('Вероятность p', fontsize=12)
plt.ylabel('Энтропия H(p), бит', fontsize=12)
plt.title('Энтропия бинарной случайной величины', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- Шаг 2. Таблица значений (ИСПРАВЛЕНО) ---
my_probabilities = [0.05, 0.2, 0.5, 0.8, 0.95]

print("Таблица значений энтропии:")
print("-" * 35)
for p in my_probabilities:
    H = entropy_binary(p)   # ← БЕЗ [0], просто скаляр
    print(f"p = {p:.2f}  →  H(p) = {H:.4f} бит")

# --- Шаг 3. Энтропия дискретных распределений ---
def entropy_discrete(probs):
    """Энтропия дискретного распределения в битах."""
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    mask = probs > 0
    return -np.sum(probs[mask] * np.log2(probs[mask]))

P = [0.4, 0.3, 0.2, 0.1]
Q = [0.25, 0.25, 0.25, 0.25]

H_P = entropy_discrete(P)
H_Q = entropy_discrete(Q)

print(f"\nРаспределение P = {P}")
print(f"H(P) = {H_P:.4f} бит")
print(f"\nРаспределение Q = {Q}")
print(f"H(Q) = {H_Q:.4f} бит")
print(f"\n➡ Большая энтропия у: {'Q (равномерное)' if H_Q > H_P else 'P'}")
