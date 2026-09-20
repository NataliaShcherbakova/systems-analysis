"""
Практика 14: Агентная модель клеточной дифференцировки
"""

import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


class CellAgent(Agent):
    """Агент-клетка с нелинейной самоорганизацией."""

    def __init__(self, model, num_genes=3):
        super().__init__(model)
        self.gene_expression = np.random.uniform(0, 1, num_genes)
        self.num_genes = num_genes

    def step(self):
        """Обновление экспрессии с нелинейной самоорганизацией."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        if len(neighbors) > 0:
            neighbor_expressions = np.array([
                agent.gene_expression for agent in neighbors
            ])
            mean_neighbor = np.mean(neighbor_expressions, axis=0)
        else:
            mean_neighbor = self.gene_expression

        alpha = self.model.alpha
        beta = self.model.beta
        sigma = self.model.sigma

        current = self.gene_expression

        # 1. Взаимодействие с соседями
        interaction = alpha * (mean_neighbor - current)

        # 2. Нелинейное самоусиление (бистабильность)
        nonlinear_boost = 2.0 * alpha * (current - 0.5) * current * (1 - current)

        # 3. Деградация к среднему
        decay = -beta * (current - 0.5)

        # 4. Стохастический шум
        noise = np.random.normal(0, sigma, self.num_genes)

        # Обновление экспрессии
        self.gene_expression = np.clip(
            current + interaction + nonlinear_boost + decay + noise,
            0, 1
        )


class CellDifferentiationModel(Model):
    """Агентная модель клеточной дифференцировки."""

    def __init__(self, N=20, num_cells=200, alpha=0.0, beta=0.1, sigma=0.05):
        super().__init__()

        self.num_cells = num_cells
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.grid = MultiGrid(N, N, torus=False)

        for i in range(num_cells):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = CellAgent(self, num_genes=3)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "mean_expression": lambda m: np.mean([
                    a.gene_expression for a in m.agents
                ], axis=0)
            }
        )

    def step(self):
        """Один шаг модели."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

    def get_all_expressions(self):
        """Получить матрицу экспрессии всех клеток."""
        return np.array([agent.gene_expression for agent in self.agents])


def evaluate_clustering(expressions, max_k=8):
    """Оценка качества кластеризации двумя методами."""
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)

    inertias = []
    sil_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(expressions_scaled)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(expressions_scaled, labels))

    # Визуализация обоих методов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Метод локтя
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Количество кластеров (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.set_title('Метод локтя (Elbow Method)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Silhouette Score
    ax2.plot(K_range, sil_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Количество кластеров (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Оценка силуэта (Silhouette Score)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Оптимальное k по максимуму Silhouette Score
    optimal_k = K_range[np.argmax(sil_scores)]
    return optimal_k, sil_scores[np.argmax(sil_scores)]


def visualize_umap(expressions, title, cluster_labels):
    """Визуализация UMAP-проекции с кластерами."""
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)
    embedding = umap.UMAP(
        n_neighbors=15, min_dist=0.1, random_state=42
    ).fit_transform(expressions_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=cluster_labels, cmap='tab10',
        s=40, alpha=0.7, edgecolors='k', linewidth=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_grid(model, step_num, alpha):
    """Визуализация клеточной сетки."""
    grid_data = np.zeros((model.grid.width, model.grid.height, 3))

    for agent in model.agents:
        x, y = agent.pos
        total = np.sum(agent.gene_expression) + 1e-10
        grid_data[x, y] = agent.gene_expression / total

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_data, origin='lower')
    ax.set_title(
        f'Клеточная сетка (α={alpha}, шаг {step_num})\n'
        'RGB = экспрессия генов (R=g1, G=g2, B=g3)',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()


# ============================================================================
# ЗАПУСК ЭКСПЕРИМЕНТОВ
# ============================================================================

N = 20
num_cells = 200
num_steps = 100

results = {}

for alpha_val in [0.0, 0.5]:
    print(f"\n{'=' * 60}")
    print(f"ЭКСПЕРИМЕНТ: α = {alpha_val}")
    print(f"{'=' * 60}")

    # Создание и запуск модели
    model = CellDifferentiationModel(
        N=N, num_cells=num_cells,
        alpha=alpha_val, beta=0.1, sigma=0.05
    )

    print(f"Запуск модели на {num_steps} шагов...")
    for _ in range(num_steps):
        model.step()

    expressions = model.get_all_expressions()

    # 1. Оценка кластеризации
    optimal_k, best_sil = evaluate_clustering(expressions, max_k=8)
    print(f"→ Оптимальное число кластеров: {optimal_k}")
    print(f"→ Silhouette Score: {best_sil:.3f}")

    results[alpha_val] = {
        'optimal_k': optimal_k,
        'silhouette': best_sil,
        'expressions': expressions
    }

    # 2. Визуализация UMAP
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(expressions_scaled)

    visualize_umap(
        expressions,
        f'UMAP: α = {alpha_val} (Кластеров: {optimal_k})',
        labels
    )

    # 3. Визуализация сетки
    visualize_grid(model, num_steps, alpha_val)

# ============================================================================
# СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

print(f"\n{'=' * 60}")
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print(f"{'=' * 60}")

score_0 = results[0.0]['silhouette']
score_05 = results[0.5]['silhouette']
k_0 = results[0.0]['optimal_k']
k_05 = results[0.5]['optimal_k']

improvement = ((score_05 / score_0) - 1) * 100 if score_0 > 0 else 0
is_emergent = score_05 > score_0 * 1.3

print(f"""
При α = 0 (нет взаимодействия):
  • Оптимальное число кластеров: {k_0}
  • Silhouette Score: {score_0:.3f}
  • Интерпретация: {'Случайное распределение' if score_0 < 0.3 else 'Слабая структура'}

При α = 0.5 (сильное взаимодействие):
  • Оптимальное число кластеров: {k_05}
  • Silhouette Score: {score_05:.3f}
  • Интерпретация: {'Эмерджентная самоорганизация!' if score_05 > 0.4 else 'Умеренная структура'}

Изменение Silhouette Score: +{improvement:.1f}%
Изменение числа кластеров: {k_0} → {k_05}

Вывод: {'✅ Модель демонстрирует эмерджентность! Клетки самоорганизуются в устойчивые аттракторы.' if is_emergent else '⚠️ Модель нуждается в доработке'}
""")
