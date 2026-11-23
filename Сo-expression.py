# ============================================================
# ПРАКТИЧЕСКОЕ ЗАДАНИЕ: Анализ сети ко-экспрессии генов
# Дисциплина: Системный анализ в биоинформатике
# Лекция 13: Сложные сети и теория графов
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# ПОДГОТОВКА: Генерация синтетических данных RNA-seq (Ground Truth)
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
print("\n" + "=" * 60)
print("ЭТАП 1. ПОСТРОЕНИЕ СЕТИ КО-ЭКСПРЕССИИ")
print("=" * 60)

corr_matrix = df_expr.T.corr(method='pearson')

# 🚨 ЗАДАНИЕ (Анализ чувствительности):
# 1. Запустите код со значением threshold_r = 0.60. Запомните число рёбер, плотность и LCC.
# 2. Измените threshold_r на 0.2 (слишком мягкий порог). Запустите снова.
#    → Что произошло с графом? (подсказка: "волосатость", hairball)
# 3. Измените threshold_r на 0.8 (слишком жёсткий порог). Запустите снова.
#    → Что произошло с LCC? Почему сеть распалась?
# 4. Ответьте: как порог влияет на "волосатость" графа и размер LCC?
#    Почему существует "оптимальный" порог?
# ⚠️ Для финальных Этапов 2-5 верните значение 0.60!

threshold_r = 0.60  # <-- ИЗМЕНИТЕ ЭТО ЗНАЧЕНИЕ для эксперимента
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
lcc_size = len(max(nx.connected_components(G), key=len))
print(f"  • Размер наибольшей связной компоненты (LCC): {lcc_size} из {G.number_of_nodes()}")

# ============================================================
# ЭТАП 2. Метрики центральности и их анализ
# ============================================================
print("\n" + "=" * 60)
print("ЭТАП 2. МЕТРИКИ ЦЕНТРАЛЬНОСТИ")
print("=" * 60)

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
print(metrics_df.sort_values('Degree_centrality', ascending=False).head(5).to_string())
print("\nТоп-5 генов по Betweenness Centrality:")
print(metrics_df.sort_values('Betweenness', ascending=False).head(5).to_string())

# 🚨 ЗАДАНИЕ (Анализ центральности):
# 1. Найдите ген с максимальной betweenness centrality.
#    К какому модулю Louvain он принадлежит? (см. Этап 3)
# 2. Почему он оказался "мостом"? Посмотрите на его связи в визуализации (Этап 5).
# 3. Совпадает ли топ-5 по degree с топ-5 по betweenness?
#    Если нет — объясните, почему локальная популярность (degree) не всегда
#    означает глобальное влияние (betweenness).

# ============================================================
# ЭТАП 3. Выявление модулей и валидация
# ============================================================
print("\n" + "=" * 60)
print("ЭТАП 3. ВЫЯВЛЕНИЕ МОДУЛЕЙ И ВАЛИДАЦИЯ")
print("=" * 60)

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
print("(ARI = 1 — полное совпадение, ARI = 0 — случайное разбиение)")

# Распределение степеней
degrees = [d for n, d in G.degree()]
degree_counts = pd.Series(degrees).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(degree_counts.index, degree_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Степень вершины k', fontsize=11)
axes[0].set_ylabel('Число вершин', fontsize=11)
axes[0].set_title('Распределение степеней (линейный)', fontsize=12)
axes[0].axvline(np.mean(degrees), color='red', linestyle='--', label=f'Среднее = {np.mean(degrees):.2f}')
axes[0].legend(fontsize=9)

log_k = np.log10(degree_counts.index)
log_P = np.log10(degree_counts.values / sum(degree_counts.values))
axes[1].scatter(degree_counts.index, degree_counts.values / sum(degree_counts.values), color='darkred', s=50, alpha=0.7)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel('Степень k (log)', fontsize=11)
axes[1].set_ylabel('P(k) (log)', fontsize=11)
axes[1].set_title('Распределение степеней (log-log)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# ✅ ИСПРАВЛЕНИЕ: инициализация gamma_estimated значением по умолчанию
gamma_estimated = None

if len(log_k) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_P)
    gamma_estimated = -slope
    axes[1].plot(degree_counts.index, 10 ** (intercept + slope * log_k), 'k--', alpha=0.5,
                 label=f'γ ≈ {gamma_estimated:.2f}')
    axes[1].legend(fontsize=9)
    print(f"\nОценённый показатель степенного закона: γ = {gamma_estimated:.2f}")
else:
    print(f"\n⚠️ Невозможно оценить γ: все узлы имеют одинаковую степень ({int(degree_counts.index[0])})")
    print("   Это характерно для сети, распавшейся на изолированные клики.")

plt.tight_layout()
plt.savefig('degree_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 🚨 ЗАДАНИЕ (Анализ распределения и валидация):
# 1. Посмотрите на значение γ (гамма). Попадает ли оно в диапазон 2.0-3.0?
#    Если нет, как вы думаете, почему? (Подсказка: посмотрите на плотность сети из Этапа 1).
# 2. При эксперименте с порогами (Этап 1) вы можете получить ARI < 0.7.
#    В этом случае предложите гипотезу, почему алгоритм не справился
#    (слишком много шума? слишком жёсткий порог? модули слились?),
#    и проверьте её, изменив порог.

# ============================================================
# ЭТАП 4. Устойчивость сети (Robustness)
# ============================================================
print("\n" + "=" * 60)
print("ЭТАП 4. ЭКСПЕРИМЕНТ ПО УСТОЙЧИВОСТИ СЕТИ")
print("=" * 60)

G_random = G.copy()
G_targeted = G.copy()

# 🚨 ЗАДАНИЕ (Устойчивость сети):
# 1. Посмотрите на процент потери при случайном удалении (5 узлов)
#    и при атаке на хабы (15 узлов).
# 2. Подтверждается ли на ваших данных принцип
#    "устойчивость к случайным отказам, уязвимость к атакам на хабы"?
# 3. Объясните: почему плотные сети более устойчивы к точечным атакам?

# Случайное удаление 5 вершин (имитация случайных мутаций)
N_RANDOM = 5
nodes_to_remove_random = np.random.choice(list(G_random.nodes()), size=N_RANDOM, replace=False)
print(f"\n1. СЛУЧАЙНОЕ УДАЛЕНИЕ (имитация случайных мутаций):")
print(f"   Удаляем {N_RANDOM} случайных узлов")
G_random.remove_nodes_from(nodes_to_remove_random)

# Целенаправленное удаление 15 вершин с максимальной degree (атака на хабы)
N_HUBS = 15
top_hubs = metrics_df.sort_values('Degree', ascending=False).head(N_HUBS).index.tolist()
print(f"\n2. ЦЕЛЕНАПРАВЛЕННОЕ УДАЛЕНИЕ (атака на хабы):")
print(f"   Удаляем топ-{N_HUBS} хабов")
G_targeted.remove_nodes_from(top_hubs)

lcc_original = len(max(nx.connected_components(G), key=len))
lcc_random = len(max(nx.connected_components(G_random), key=len)) if nx.number_of_nodes(G_random) > 0 else 0
lcc_targeted = len(max(nx.connected_components(G_targeted), key=len)) if nx.number_of_nodes(G_targeted) > 0 else 0

print("\n" + "-" * 60)
print("РЕЗУЛЬТАТЫ:")
print(f"Исходный размер наибольшей компоненты (LCC): {lcc_original} генов")
print(
    f"После случайного удаления {N_RANDOM} узлов: {lcc_random} генов (потеря: {(lcc_original - lcc_random) / lcc_original * 100:.1f}%)")
print(
    f"После удаления {N_HUBS} хабов: {lcc_targeted} генов (потеря: {(lcc_original - lcc_targeted) / lcc_original * 100:.1f}%)")

ratio = (lcc_original - lcc_targeted) / (lcc_original - lcc_random) if (lcc_original - lcc_random) > 0 else float('inf')
print(f"\nОтношение ущерба (атака/случайность): {'∞' if ratio == float('inf') else f'{ratio:.2f}'}")

if ratio > 3:
    print("✅ ГИПОТЕЗА ПОДТВЕРДИЛАСЬ: атака на хабы нанесла значительно больший ущерб.")
elif ratio > 1.5:
    print("⚠️ ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРДИЛАСЬ.")
else:
    print("❌ ГИПОТЕЗА НЕ ПОДТВЕРДИЛАСЬ: сеть слишком плотная и устойчивая.")

# ============================================================
# ЭТАП 5. Визуализация и выводы
# ============================================================
print("\n" + "=" * 60)
print("ЭТАП 5. ВИЗУАЛИЗАЦИЯ И ВЫВОДЫ")
print("=" * 60)

# 5.1. Визуализация сети с модулями
plt.figure(figsize=(12, 10))

pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

cmap = plt.cm.tab10
node_colors = [cmap(partition[n]) for n in G.nodes()]

max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
node_sizes = [150 + 400 * (G.degree(n) / max_deg) for n in G.nodes()]

nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.8, edge_color='lightgray')
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       alpha=0.95, edgecolors='white', linewidths=0.8)

top_hubs_list = metrics_df.sort_values('Degree', ascending=False).head(5).index.tolist()
top_bridges_list = metrics_df.sort_values('Betweenness', ascending=False).head(3).index.tolist()
labels_to_show = list(set(top_hubs_list + top_bridges_list))

nx.draw_networkx_labels(G, pos, labels={n: n for n in labels_to_show},
                        font_size=8, font_weight='bold', font_color='black',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(i), markersize=10, alpha=0.95,
                          label=f'Модуль {i}')
                   for i in range(n_communities)]
plt.legend(handles=legend_elements, loc='upper right', title='Модули Louvain',
           fontsize=10, title_fontsize=11, framealpha=0.9)

plt.title(f"Сеть ко-экспрессии: {n_communities} модулей, Q={modularity:.3f}, "
          f"ARI={ari:.3f}, thr={threshold_r:.2f}", fontsize=13, fontweight='bold', pad=15)
plt.axis('off')
plt.tight_layout()
plt.savefig('coexpression_network.png', dpi=200, bbox_inches='tight')
plt.show()

# 5.2. Выявление терапевтических мишеней (с корректным маппингом модулей)
true_disease_labels = [0, 1]

louvain_disease_labels = set()
for gene, true_mod in true_labels.items():
    if true_mod in true_disease_labels:
        louvain_disease_labels.add(partition[gene])

print(f"\nИстинные модули заболевания: {true_disease_labels}")
print(f"Алгоритм Louvain присвоил им следующие метки: {louvain_disease_labels}")

module_membership = partition
disease_module_genes = [gene for gene, mod in module_membership.items()
                        if mod in louvain_disease_labels]
disease_hubs = sorted(disease_module_genes, key=lambda x: G.degree(x), reverse=True)

print("\n🎯 ТОП-5 ГЕНОВ-ХАБОВ В 'ЗАБОЛЕВАНИИ' (корректно найденные модули):")
print("-" * 60)
for i, gene in enumerate(disease_hubs[:5], 1):
    degree = G.degree(gene)
    betweenness = betweenness_cent[gene]
    louvain_mod = module_membership[gene]
    true_mod = true_labels[gene]

    print(f"{i}. {gene} (Louvain модуль: {louvain_mod}, Истинный модуль: {true_mod})")
    print(f"   • Степень (degree): {degree}")
    print(f"   • Betweenness centrality: {betweenness:.4f}")

    if betweenness > np.percentile(list(betweenness_cent.values()), 90):
        print(f"   • 🌉 Это МОСТ между модулями (топ-10% по betweenness)")
    if degree > np.percentile([G.degree(n) for n in G.nodes()], 90):
        print(f"   • 🎯 Это ХАБ сети (топ-10% по степени)")

# 🚨 ЗАДАНИЕ (Биологическая интерпретация):
# Сформулируйте 2-3 предложения:
# 1. Какие модули ассоциированы с заболеванием?
# 2. Какие конкретные гены являются кандидатами на роль терапевтических мишеней?
# 3. Почему именно они? (опирайтесь на их метрики центральности и принадлежность к модулю)

# Итоговые выводы
print("\n" + "=" * 60)
print("ИТОГОВЫЕ ВЫВОДЫ АНАЛИЗА")
print("=" * 60)

print(f"\n1. Структура сети:")
print(f"   • Число вершин: {G.number_of_nodes()}")
print(f"   • Число рёбер: {G.number_of_edges()}")
print(f"   • Средняя степень: {np.mean(degrees):.2f}")
print(f"   • Плотность графа: {nx.density(G):.4f}")
print(f"   • Средний кластеризационный коэффициент: {avg_clustering:.4f}")

print(f"\n2. Топ-3 хаба (кандидаты на роль ключевых регуляторов):")
for gene in metrics_df.sort_values('Degree_centrality', ascending=False).head(3).index:
    print(f"   • {gene}: degree={G.degree(gene)}, betweenness={betweenness_cent[gene]:.4f}")

print(f"\n3. Качество кластеризации:")
print(f"   • Число сообществ: {n_communities}")
print(f"   • Модульность Q: {modularity:.4f}")
print(f"   • ARI с истинными модулями: {ari:.4f}")
# ✅ ИСПРАВЛЕНИЕ: безопасный вывод gamma_estimated
if gamma_estimated is not None:
    print(f"   • Показатель степенного закона: γ = {gamma_estimated:.2f}")
else:
    print(f"   • Показатель степенного закона: не определён (однородная сеть)")

print(f"\n4. Устойчивость сети:")
print(
    f"   • Потеря LCC при случайном удалении {N_RANDOM} узлов: {(lcc_original - lcc_random) / lcc_original * 100:.1f}%")
print(f"   • Потеря LCC при атаке на {N_HUBS} хабов: {(lcc_original - lcc_targeted) / lcc_original * 100:.1f}%")
print(f"   • Отношение ущерба (атака/случайность): {'∞' if ratio == float('inf') else f'{ratio:.2f}'}")

print("\n" + "=" * 60)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("Графики сохранены как 'degree_distribution.png' и 'coexpression_network.png'")
print("=" * 60)
