# ============================================================
# ЗАДАНИЕ 5. Permutation test и сеть коэкспрессии
# ============================================================

# 1. ИМПОРТЫ (Обязательно в начале каждого отдельного .py файла!)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore')

# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (Обязательно, так как это отдельный файл!)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X_log = np.log1p(X)  # логарифмирование для стабилизации дисперсии

gene_names = [
    'Ген A\n(sepal length)',
    'Ген B\n(sepal width)',
    'Ген C\n(petal length)',
    'Ген D\n(petal width)'
]
short_names = ['Ген A', 'Ген B', 'Ген C', 'Ген D']

# 3. ПАРАМЕТРЫ ТЕСТА (ИЗМЕНИТЕ при желании)
N_PERMUTATIONS = 500  # число пермутаций (больше = точнее, но медленнее)
FDR_THRESHOLD = 0.05  # порог значимости


# 4. ФУНКЦИЯ PERMUTATION TEST (НЕ МЕНЯЙТЕ)
def permutation_test_mi(x, y, n_permutations=N_PERMUTATIONS, random_state=42):
    rng = np.random.default_rng(random_state)

    # Реальное MI
    mi_real = mutual_info_regression(x.reshape(-1, 1), y, random_state=random_state)[0]

    # Null-распределение
    null_dist = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)  # перемешиваем y
        mi_perm = mutual_info_regression(x.reshape(-1, 1), y_perm, random_state=random_state)[0]
        null_dist.append(mi_perm)

    # p-value (с поправкой +1)
    null_arr = np.array(null_dist)
    p_value = (np.sum(null_arr >= mi_real) + 1) / (n_permutations + 1)

    return mi_real, p_value, null_arr


# 5. ПРИМЕНЕНИЕ КО ВСЕМ ПАРАМ
n_genes = X_log.shape[1]
pairs, mi_values, p_values = [], [], []

print(f"Вычисляем permutation test для всех пар ({n_genes * (n_genes - 1) // 2} пар)...")
print(f"Число пермутаций: {N_PERMUTATIONS}\n")

for i in range(n_genes):
    for j in range(i + 1, n_genes):
        mi, p, _ = permutation_test_mi(
            X_log.iloc[:, i].values,
            X_log.iloc[:, j].values
        )
        pairs.append((i, j))
        mi_values.append(mi)
        p_values.append(p)

# 6. FDR-КОРРЕКЦИЯ (Benjamini-Hochberg)
reject, p_adj, _, _ = multipletests(p_values, alpha=FDR_THRESHOLD, method='fdr_bh')

print("=== Результаты для всех пар ===")
print(f"{'Пара':<20} {'MI':<10} {'p-value':<12} {'p_adj (FDR)':<12} {'Значима?'}")
print("-" * 70)
for (i, j), mi, p, pa, rej in zip(pairs, mi_values, p_values, p_adj, reject):
    name_i = gene_names[i].replace('\n', ' ')
    name_j = gene_names[j].replace('\n', ' ')
    mark = '✓ ДА' if rej else '✗ нет'
    print(f"{name_i} — {name_j:<5} {mi:<10.4f} {p:<12.4f} {pa:<12.4f} {mark}")

# 7. ГИСТОГРАММА NULL-РАСПРЕДЕЛЕНИЯ (для самой значимой пары)
best_idx = np.argmin(p_adj)
best_i, best_j = pairs[best_idx]
_, _, best_null = permutation_test_mi(
    X_log.iloc[:, best_i].values,
    X_log.iloc[:, best_j].values,
    n_permutations=1000  # для красивой гистограммы берем 1000
)
best_mi = mi_values[best_idx]
threshold = np.percentile(best_null, 95)

fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(best_null, bins=30, color='lightgray', edgecolor='black', stat='count', ax=ax)

# Динамическое позиционирование
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.4)

ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'95-й перцентиль ({threshold:.3f})')
ax.axvline(best_mi, color='red', linestyle='-', linewidth=2.5, label=f'Наблюдаемое MI = {best_mi:.3f}')

ax.text(best_mi, y_max * 1.25, f'Наблюдаемое\nMI = {best_mi:.3f}',
        color='red', fontsize=10, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.95))

ax.set_xlabel('Взаимная информация, MI (бит)', fontsize=12)
ax.set_ylabel('Частота (из 1000 пермутаций)', fontsize=12)
name_i = gene_names[best_i].replace('\n', ' ')
name_j = gene_names[best_j].replace('\n', ' ')
ax.set_title(f'Null-распределение MI: {name_i} — {name_j}', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()

# 8. ПОСТРОЕНИЕ ГРАФА СЕТИ
G = nx.Graph()
G.add_nodes_from(short_names)

# Добавляем только значимые рёбра (p_adj < 0.05)
for (i, j), mi, pa, rej in zip(pairs, mi_values, p_adj, reject):
    if rej:
        G.add_edge(short_names[i], short_names[j], weight=mi)

plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42, k=0.5)

# Размер узла пропорционален degree
degrees = dict(G.degree())
node_sizes = [degrees[n] * 1500 + 1000 for n in G.nodes()]

# Толщина ребра пропорциональна MI
edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', edgecolors='black', linewidths=2)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='steelblue', alpha=0.7)

plt.title(f'Сеть коэкспрессии (FDR < {FDR_THRESHOLD})', fontsize=13)
plt.axis('off')
plt.tight_layout()
plt.show()

# 9. МЕТРИКИ СЕТИ
print("\n=== Метрики сети ===")
print(f"Число узлов: {G.number_of_nodes()}")
print(f"Число рёбер: {G.number_of_edges()}")
print(f"Плотность графа: {nx.density(G):.3f}")

print("\nСтепень каждого гена:")
for node in G.nodes():
    print(f"  {node}: degree = {degrees[node]}")

hub_gene = max(degrees, key=degrees.get)
print(f"\n🏆 Hub-ген (максимальная степень): {hub_gene} (degree = {degrees[hub_gene]})")