# ============================================================
# ПРАКТИЧЕСКОЕ ЗАДАНИЕ: Анализ сети ко-экспрессии генов
# Дисциплина: Системный анализ в биоинформатике
# Лекция 13: Сложные сети и теория графов
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# ПОДГОТОВКА: Генерация синтетических данных RNA-seq
# ============================================================
n_genes = 100
n_samples = 40
gene_names = [f"GENE_{i + 1:03d}" for i in range(n_genes)]

n_modules = 5
module_size = n_genes // n_modules
true_modules = {}

expression_matrix = np.zeros((n_genes, n_samples))
global_signal = np.random.randn(n_samples) * 1.5

for m in range(n_modules):
    start_idx = m * module_size
    end_idx = start_idx + module_size
    module_genes = gene_names[start_idx:end_idx]
    true_modules[m] = module_genes

    base_signal = np.random.randn(n_samples) * 2.0 + 10

    for i, gene in enumerate(module_genes):
        noise = np.random.randn(n_samples) * 0.5
        expression_matrix[start_idx + i, :] = base_signal + noise + global_signal

        if m in [0, 1]:
            expression_matrix[start_idx + i, 20:] += 4.0

df_expr = pd.DataFrame(expression_matrix, index=gene_names,
                       columns=[f"Sample_{i + 1}" for i in range(n_samples)])

print(f"ПОДГОТОВКА: Матрица экспрессии {df_expr.shape[0]} генов × {df_expr.shape[1]} образцов")
print(f"Модули 0 и 1 активированы в 'заболевании' (образцы 21-40)")

# ============================================================
# ЭТАП 1. Построение сети ко-экспрессии
# ============================================================
print("\n" + "="*60)
print("ЭТАП 1. ПОСТРОЕНИЕ СЕТИ КО-ЭКСПРЕССИИ")
print("="*60)

corr_matrix = df_expr.T.corr(method='pearson')

# 🚨 ЗАДАНИЕ (Анализ чувствительности):
# 1. Запустите код со значением threshold_r = 0.35. Запомните число рёбер и плотность.
# 2. Измените threshold_r на 0.2 (слишком мягкий порог). Запустите снова.
# 3. Измените threshold_r на 0.5 (слишком жёсткий порог). Запустите снова.
# 4. Ответьте: как порог влияет на "волосатость" графа и размер наибольшей компоненты (LCC)?
# ⚠️ Для финальной визуализации и этапов 2-5 верните значение 0.35!

threshold_r = 0.35  # <-- ИЗМЕНИТЕ ЭТО ЗНАЧЕНИЕ для эксперимента
threshold_p = 0.05

G = nx.Graph()
G.add_nodes_from(gene_names)

edges_added = 0
for i in range(n_genes):
    for j in range(i + 1, n_genes):
        gene_i = gene_names[i]
        gene_j = gene_names[j]
        r = corr_matrix.loc[gene_i, gene_j]
        r_clean, p_value = stats.pearsonr(df_expr.loc[gene_i], df_expr.loc[gene_j])

        if abs(r) > threshold_r and p_value < threshold_p:
            G.add_edge(gene_i, gene_j, weight=r)
            edges_added += 1

print(f"\nПостроена сеть ко-экспрессии (при пороге |r|>{threshold_r}):")
print(f"  • Вершин: {G.number_of_nodes()}")
print(f"  • Рёбер: {G.number_of_edges()}")
print(f"  • Средняя степень: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"  • Плотность графа: {nx.density(G):.4f}")
print(f"  • Размер наибольшей связной компоненты: {len(max(nx.connected_components(G), key=len))} из {G.number_of_nodes()}")

# ============================================================
# ЭТАП 2. Метрики центральности
# ============================================================
print("\n" + "="*60)
print("ЭТАП 2. МЕТРИКИ ЦЕНТРАЛЬНОСТИ")
print("="*60)

degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight='weight')
closeness_cent = nx.closeness_centrality(G)

try:
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
except:
    eigenvector_cent = {n: 0 for n in G.nodes()}

clustering_coeff = nx.clustering(G)
avg_clustering = nx.average_clustering(G)

metrics_df = pd.DataFrame({
    'Degree': [G.degree(n) for n in gene_names],
    'Degree_centrality': [degree_cent[n] for n in gene_names],
    'Betweenness': [betweenness_cent[n] for n in gene_names],
    'Closeness': [closeness_cent[n] for n in gene_names],
    'Eigenvector': [eigenvector_cent[n] for n in gene_names],
    'Clustering': [clustering_coeff[n] for n in gene_names]
}, index=gene_names)

print(f"\nСредний кластеризационный коэффициент: {avg_clustering:.4f}")
print("\nТоп-5 генов по Degree Centrality:")
print(metrics_df.sort_values('Degree_centrality', ascending=False).head(5))
print("\nТоп-5 генов по Betweenness Centrality:")
print(metrics_df.sort_values('Betweenness', ascending=False).head(5))

# 🚨 ЗАДАНИЕ (Анализ центральности):
# 1. Сравните Топ-5 генов по Degree и по Betweenness.
# 2. Найдите ген, который есть в списке Betweenness, но отсутствует в списке Degree.
# 3. Посмотрите на визуализацию сети (этап 5): является ли этот ген "мостом" между цветными кластерами?

# ============================================================
# ЭТАП 3. Выявление модулей и валидация
# ============================================================
print("\n" + "="*60)
print("ЭТАП 3. ВЫЯВЛЕНИЕ МОДУЛЕЙ И ВАЛИДАЦИЯ")
print("="*60)

partition = community_louvain.best_partition(G, weight='weight')
n_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G, weight='weight')

print(f"\nАлгоритм Louvain выявил {n_communities} сообществ")
print(f"Модульность Q = {modularity:.4f}")

true_labels = {}
for m, genes in true_modules.items():
    for g in genes:
        true_labels[g] = m

ari = adjusted_rand_score([true_labels[n] for n in G.nodes()], [partition[n] for n in G.nodes()])
print(f"\nAdjusted Rand Index (сравнение с истинными модулями): {ari:.4f}")

# Распределение степеней
degrees = [d for n, d in G.degree()]
degree_counts = pd.Series(degrees).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(degree_counts.index, degree_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Степень вершины k', fontsize=12)
axes[0].set_ylabel('Число вершин', fontsize=12)
axes[0].set_title('Распределение степеней (линейный масштаб)', fontsize=13)
axes[0].axvline(np.mean(degrees), color='red', linestyle='--', label=f'Среднее = {np.mean(degrees):.2f}')
axes[0].legend()

log_k = np.log10(degree_counts.index)
log_P = np.log10(degree_counts.values / sum(degree_counts.values))
axes[1].scatter(degree_counts.index, degree_counts.values / sum(degree_counts.values), color='darkred', s=50, alpha=0.7)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Степень k (log)', fontsize=12)
axes[1].set_ylabel('P(k) (log)', fontsize=12)
axes[1].set_title('Распределение степеней (log-log масштаб)', fontsize=13)
axes[1].grid(True, alpha=0.3)

if len(log_k) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_P)
    gamma_estimated = -slope
    axes[1].plot(degree_counts.index, 10**(intercept + slope * log_k), 'k--', alpha=0.5, label=f'γ ≈ {gamma_estimated:.2f}')
    axes[1].legend()
    print(f"\nОценённый показатель степенного закона: γ = {gamma_estimated:.2f}")

plt.tight_layout()
plt.savefig('degree_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 🚨 ЗАДАНИЕ (Анализ распределения и валидация):
# 1. Посмотрите на значение γ (гамма). Попадает ли оно в диапазон 2.0-3.0?
#    Если нет, как вы думаете, почему? (Подсказка: посмотрите на плотность сети из Этапа 1).
# 2. Посмотрите на ARI. Если он меньше 0.7, попробуйте изменить threshold_r в Этапе 1,
#    чтобы улучшить совпадение выявленных модулей с истинными.

# ============================================================
# ЭТАП 4. Устойчивость сети (Robustness)
# ============================================================
print("\n" + "="*60)
print("ЭТАП 4. ЭКСПЕРИМЕНТ ПО УСТОЙЧИВОСТИ СЕТИ")
print("="*60)

G_random = G.copy()
G_targeted = G.copy()

# 🚨 ЗАДАНИЕ (Устойчивость сети):
# 1. Посмотрите на процент потери при случайном удалении и при атаке на хабы.
# 2. Если гипотеза не подтвердилась (потери одинаковые), измените в коде size=5 на size=15
#    и head(5) на head(15) в блоке целенаправленного удаления. Запустите снова.
# 3. Объясните: почему плотные сети более устойчивы к точечным атакам?

nodes_to_remove_random = np.random.choice(list(G_random.nodes()), size=5, replace=False) # <-- Попробуйте изменить size
print(f"\n1. СЛУЧАЙНОЕ УДАЛЕНИЕ (имитация случайных мутаций):")
print(f"   Удаляем случайные узлы: {nodes_to_remove_random}")
G_random.remove_nodes_from(nodes_to_remove_random)

top_hubs = metrics_df.sort_values('Degree', ascending=False).head(5).index.tolist() # <-- Попробуйте изменить head(5) на head(15)
print(f"\n2. ЦЕЛЕНАПРАВЛЕННОЕ УДАЛЕНИЕ (атака на хабы):")
print(f"   Удаляем топ-5 хабов: {top_hubs}")
G_targeted.remove_nodes_from(top_hubs)

lcc_original = len(max(nx.connected_components(G), key=len))
lcc_random = len(max(nx.connected_components(G_random), key=len)) if nx.number_of_nodes(G_random) > 0 else 0
lcc_targeted = len(max(nx.connected_components(G_targeted), key=len)) if nx.number_of_nodes(G_targeted) > 0 else 0

print("\n" + "-"*60)
print("РЕЗУЛЬТАТЫ:")
print(f"Исходный размер наибольшей компоненты: {lcc_original} генов")
print(f"После случайного удаления: {lcc_random} генов (потеря: {(lcc_original - lcc_random)/lcc_original*100:.1f}%)")
print(f"После удаления хабов: {lcc_targeted} генов (потеря: {(lcc_original - lcc_targeted)/lcc_original*100:.1f}%)")

ratio = (lcc_original - lcc_targeted) / (lcc_original - lcc_random) if (lcc_original - lcc_random) > 0 else float('inf')
print(f"\nОтношение ущерба (атака/случайность): {'∞' if ratio == float('inf') else f'{ratio:.2f}'}")

if ratio > 3: print("✅ ГИПОТЕЗА ПОДТВЕРДИЛАСЬ: атака на хабы нанесла значительно больший ущерб.")
elif ratio > 1.5: print("️ ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРДИЛАСЬ.")
else: print("❌ ГИПОТЕЗА НЕ ПОДТВЕРДИЛАСЬ: сеть слишком плотная и устойчивая.")

# ============================================================
# ЭТАП 5. Визуализация и выводы
# ============================================================
print("\n" + "="*60)
print("ЭТАП 5. ВИЗУАЛИЗАЦИЯ И ВЫВОДЫ")
print("="*60)

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

cmap = plt.cm.tab10
node_colors = [partition[n] for n in G.nodes()]
node_sizes = [300 + 2000 * degree_cent[n] for n in G.nodes()]

nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8)

top5_betweenness = metrics_df.sort_values('Betweenness', ascending=False).head(5).index
top5_degree = metrics_df.sort_values('Degree_centrality', ascending=False).head(5).index
labels_to_show = list(set(top5_betweenness).union(set(top5_degree)))

nx.draw_networkx_labels(G, pos, labels={n: n for n in labels_to_show}, font_size=9, font_weight='bold', font_color='black')
plt.title(f"Сеть ко-экспрессии: {n_communities} модулей, Q={modularity:.3f}, ARI={ari:.3f}", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('coexpression_network.png', dpi=150, bbox_inches='tight')
plt.show()

module_membership = partition
disease_module_genes = [gene for gene, mod in module_membership.items() if mod in [0, 1]]
disease_hubs = sorted(disease_module_genes, key=lambda x: G.degree(x), reverse=True)

print("\n🎯 ТОП-5 ГЕНОВ-ХАБОВ В 'ЗАБОЛЕВАНИИ' (модули 0 и 1):")
print("-" * 60)
for i, gene in enumerate(disease_hubs[:5], 1):
    degree = G.degree(gene)
    betweenness = betweenness_cent[gene]
    module = module_membership[gene]
    print(f"{i}. {gene} (Модуль: {module}, Degree: {degree}, Betweenness: {betweenness:.4f})")

    if betweenness > np.percentile(list(betweenness_cent.values()), 90):
        print(f"   • 🌉 Это МОСТ между модулями (топ-10% по betweenness)")
    if degree > np.percentile([G.degree(n) for n in G.nodes()], 90):
        print(f"   • 🎯 Это ХАБ сети (топ-10% по степени)")

# 🚨 ЗАДАНИЕ (Выбор мишени):
# 1. Посмотрите на список "ТОП-5 ГЕНОВ-ХАБОВ В 'ЗАБОЛЕВАНИИ'".
# 2. Выберите один ген, который помечен и как ХАБ (🎯), и как МОСТ ().
# 3. Почему именно его стоит проверить как мишень для терапии в первую очередь?

print("\n" + "="*60)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("="*60)