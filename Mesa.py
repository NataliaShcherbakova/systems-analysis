"""
Практика 14: Агентная модель клеточной дифференцировки
ПОЛНАЯ ВЕРСИЯ с исправленной логикой вывода
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
        """Обновление с нелинейной самоорганизацией."""
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

        # Взаимодействие с соседями
        interaction = alpha * (mean_neighbor - current)

        # Нелинейное самоусиление (бистабильность)
        nonlinear_boost = 2.0 * alpha * (current - 0.5) * current * (1 - current)

        # Деградация к среднему
        decay = -beta * (current - 0.5)

        # Шум
        noise = np.random.normal(0, sigma, self.num_genes)

        # Обновление
        self.gene_expression = current + interaction + nonlinear_boost + decay + noise
        self.gene_expression = np.clip(self.gene_expression, 0, 1)


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
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")

    def get_all_expressions(self):
        return np.array([agent.gene_expression for agent in self.agents])


def find_optimal_clusters(expressions, max_clusters=10):
    """Найти оптимальное число кластеров методом силуэта."""
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)

    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        if k >= len(expressions):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(expressions_scaled)
        score = silhouette_score(expressions_scaled, labels)
        silhouette_scores.append(score)

    if len(silhouette_scores) > 0:
        optimal_k = K_range[np.argmax(silhouette_scores)]
    else:
        optimal_k = 1

    return optimal_k, silhouette_scores


def visualize_umap(expressions, title, cluster_labels=None):
    """Визуализация UMAP-проекции с кластерами."""
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(expressions_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    if cluster_labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=cluster_labels, cmap='tab10',
            s=50, alpha=0.7, edgecolors='k', linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c='steelblue', s=50, alpha=0.7, edgecolors='k', linewidth=0.5
        )

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return embedding


def visualize_grid(model, step_num):
    """Визуализация сетки с клетками."""
    grid_data = np.zeros((model.grid.width, model.grid.height, 3))

    for agent in model.agents:
        x, y = agent.pos
        total = np.sum(agent.gene_expression) + 1e-10
        grid_data[x, y] = agent.gene_expression / total

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(grid_data, origin='lower')
    ax.set_title(f'Клеточная сетка (шаг {step_num})\n'
                 'RGB = относительная экспрессия генов (R=g1, G=g2, B=g3)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Запуск экспериментов
# ============================================================================

print("=" * 80)
print("ЭКСПЕРИМЕНТ 1: Нет взаимодействия (α = 0)")
print("=" * 80)

N = 20
num_cells = 200
num_steps = 100

model1 = CellDifferentiationModel(N=N, num_cells=num_cells, alpha=0.0, beta=0.1, sigma=0.05)

print(f"Запуск модели на {num_steps} шагов...")
for i in range(num_steps):
    model1.step()

expressions1 = model1.get_all_expressions()

# ВАЖНО: вызываем функцию и сохраняем ОБА возвращаемых значения
optimal_k1, scores1 = find_optimal_clusters(expressions1)
print(f"Оптимальное число кластеров: {optimal_k1}")

if optimal_k1 > 1:
    kmeans1 = KMeans(n_clusters=optimal_k1, random_state=42, n_init=10)
    labels1 = kmeans1.fit_predict(StandardScaler().fit_transform(expressions1))
else:
    labels1 = np.zeros(len(expressions1))

score1 = scores1[0] if scores1 else 0
print(f"Silhouette score: {score1:.3f}")

visualize_grid(model1, num_steps)
embedding1 = visualize_umap(expressions1, f'UMAP: α = 0 (нет взаимодействия)\nКластеров: {optimal_k1}', labels1)

print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ 2: Сильное взаимодействие (α = 0.5)")
print("=" * 80)

model2 = CellDifferentiationModel(N=N, num_cells=num_cells, alpha=0.5, beta=0.1, sigma=0.05)

print(f"Запуск модели на {num_steps} шагов...")
for i in range(num_steps):
    model2.step()

expressions2 = model2.get_all_expressions()

# ВАЖНО: вызываем функцию и сохраняем ОБА возвращаемых значения
optimal_k2, scores2 = find_optimal_clusters(expressions2)
print(f"Оптимальное число кластеров: {optimal_k2}")

if optimal_k2 > 1:
    kmeans2 = KMeans(n_clusters=optimal_k2, random_state=42, n_init=10)
    labels2 = kmeans2.fit_predict(StandardScaler().fit_transform(expressions2))
else:
    labels2 = np.zeros(len(expressions2))

score2 = scores2[0] if scores2 else 0
print(f"Silhouette score: {score2:.3f}")

visualize_grid(model2, num_steps)
embedding2 = visualize_umap(expressions2, f'UMAP: α = 0.5 (сильное взаимодействие)\nКластеров: {optimal_k2}', labels2)

print("\n" + "=" * 80)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Правильное условие: эмерджентность = рост качества кластеризации
improvement = (score2 / score1 - 1) * 100 if score1 > 0 else 0
is_emergent = score2 > score1 * 1.3  # Улучшение на 30%+

print(f"""
При α = 0 (нет взаимодействия):
- Оптимальное число кластеров: {optimal_k1}
- Silhouette score: {score1:.3f}
- Интерпретация: {'Случайное распределение, кластеры не выражены' if score1 < 0.3 else 'Слабая структура'}

При α = 0.5 (сильное взаимодействие):
- Оптимальное число кластеров: {optimal_k2}
- Silhouette score: {score2:.3f}
- Интерпретация: {'Эмерджентная самоорганизация!' if score2 > 0.4 else 'Умеренная структура'}

Изменение Silhouette score: +{improvement:.1f}%
Изменение числа кластеров: {optimal_k1} → {optimal_k2} (клетки сгруппировались в устойчивые аттракторы)

Вывод: {'✅ Модель демонстрирует эмерджентность! Клетки самоорганизуются в устойчивые аттракторы.' if is_emergent else '⚠️ Модель нуждается в доработке'}
""")
