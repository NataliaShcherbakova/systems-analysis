# ============================================================
# ЗАДАНИЕ 3. Сравнение MI и корреляции
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

np.random.seed(42)
N = 200
X = np.random.uniform(-3, 3, N)
noise = np.random.normal(0, 0.3, N)

# 4 типа зависимостей
data = {
    'Линейная':        (X, 2 * X + noise),
    'U-образная':      (X, X**2 + noise),
    'Синусоидальная':  (X, np.sin(2 * X) + noise),
    'Независимая':     (X, np.random.normal(0, 1, N))
}

# --- Функция вычисления метрик (НЕ МЕНЯЙТЕ) ---
def compute_metrics(x, y):
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
    return r, rho, mi

# --- Таблица результатов ---
results = {}
for name, (x, y) in data.items():
    r, rho, mi = compute_metrics(x, y)
    results[name] = {'Пирсон r': round(r, 3), 'Спирмен ρ': round(rho, 3), 'MI (бит)': round(mi, 3)}

df = pd.DataFrame(results).T
print("=== Сравнение метрик ===")
print(df)

# --- Визуализация 2×2 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, (x, y)) in zip(axes, data.items()):
    ax.scatter(x, y, alpha=0.5, s=15, color='steelblue')
    r, rho, mi = results[name]['Пирсон r'], results[name]['Спирмен ρ'], results[name]['MI (бит)']
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    text = f'r = {r}\nρ = {rho}\nMI = {mi} бит'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("\n➡ Обратите внимание: для U-образной и синусоидальной зависимостей")
print("   корреляция Пирсона ≈ 0, но MI — высокое!")