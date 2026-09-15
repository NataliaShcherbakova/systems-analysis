# ============================================================
# ЗАДАНИЕ 2. KL-дивергенция и JS-дивергенция
# ============================================================
import numpy as np
# --- Функции (НЕ МЕНЯЙТЕ) ---
def kl_divergence(P, Q):
    """KL-дивергенция D_KL(P || Q) в битах."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    P = P / P.sum()
    Q = Q / Q.sum()
    kl = 0.0
    for p, q in zip(P, Q):
        if p > 0:
            if q == 0:
                return np.inf
            kl += p * np.log2(p / q)
    return kl

def js_divergence(P, Q):
    """JS-дивергенция (симметричная версия KL)."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    P = P / P.sum()
    Q = Q / Q.sum()
    M = (P + Q) / 2
    return 0.5 * kl_divergence(P, M) + 0.5 * kl_divergence(Q, M)

# --- Шаг 1. Проверка асимметричности ---
# ИЗМЕНИТЕ: подставьте свои 2 распределения (4 числа, сумма = 1)
P = [0.4, 0.3, 0.2, 0.1]
Q = [0.5, 0.2, 0.2, 0.1]

kl_PQ = kl_divergence(P, Q)
kl_QP = kl_divergence(Q, P)

print("=== Проверка асимметричности KL ===")
print(f"P = {P}")
print(f"Q = {Q}")
print(f"\nD_KL(P || Q) = {kl_PQ:.4f} бит")
print(f"D_KL(Q || P) = {kl_QP:.4f} бит")
print(f"\n➡ Значения {'РАЗНЫЕ (асимметрично) ✓' if abs(kl_PQ - kl_QP) > 1e-6 else 'одинаковые'}")

# --- Шаг 2. JS-дивергенция ---
js_PQ = js_divergence(P, Q)
print(f"\n=== JS-дивергенция (симметричная) ===")
print(f"JS(P || Q) = {js_PQ:.4f} бит")
print(f"JS(Q || P) = {js_divergence(Q, P):.4f} бит")
print(f"➡ Значения одинаковые ✓")

# --- Шаг 3. Граничный случай ---
print("\n=== Граничный случай ===")
P_edge = [0.5, 0.5, 0.0]
Q_edge = [0.5, 0.0, 0.5]  # Q[1] = 0, но P[1] = 0.5

print(f"P = {P_edge}, Q = {Q_edge}")
print(f"D_KL(P || Q) = {kl_divergence(P_edge, Q_edge)}  (должно быть inf)")
print(f"JS(P || Q)   = {js_divergence(P_edge, Q_edge):.4f}  (конечное значение)")