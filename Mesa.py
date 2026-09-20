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
    def __init__(self, model, num_genes=3):
        super().__init__(model)
        self.gene_expression = np.random.uniform(0, 1, num_genes)
        self.num_genes = num_genes
        
    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        if len(neighbors) > 0:
            neighbor_expressions = np.array([agent.gene_expression for agent in neighbors])
            mean_neighbor = np.mean(neighbor_expressions, axis=0)
        else:
            mean_neighbor = self.gene_expression
        
        alpha, beta, sigma = self.model.alpha, self.model.beta, self.model.sigma
        current = self.gene_expression
        
        interaction = alpha * (mean_neighbor - current)
        nonlinear_boost = 2.0 * alpha * (current - 0.5) * current * (1 - current)
        decay = -beta * (current - 0.5)
        noise = np.random.normal(0, sigma, self.num_genes)
        
        self.gene_expression = np.clip(current + interaction + nonlinear_boost + decay + noise, 0, 1)

class CellDifferentiationModel(Model):
    def __init__(self, N=20, num_cells=200, alpha=0.0, beta=0.1, sigma=0.05):
        super().__init__()
        self.num_cells, self.alpha, self.beta, self.sigma = num_cells, alpha, beta, sigma
        self.grid = MultiGrid(N, N, torus=False)
        
        for i in range(num_cells):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(CellAgent(self, num_genes=3), (x, y))
        
        self.datacollector = DataCollector(model_reporters={
            "mean_expression": lambda m: np.mean([a.gene_expression for a in m.agents], axis=0)
        })
    
    def step(self):
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")
    
    def get_all_expressions(self):
        return np.array([agent.gene_expression for agent in self.agents])

def evaluate_clustering(expressions, max_k=8):
    """Оценивает качество кластеризации двумя методами"""
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
    
    # 1. Метод локтя
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Количество кластеров (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.set_title('Метод локтя (Elbow Method)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Silhouette Score
    ax2.plot(K_range, sil_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Количество кластеров (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Оценка силуэта (Silhouette Score)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Оптимальное k выбираем по максимуму Silhouette Score (это надежнее, чем "локоть" на глаз)
    optimal_k = K_range[np.argmax(sil_scores)]
    return optimal_k, sil_scores[np.argmax(sil_scores)]

def visualize_umap(expressions, title, cluster_labels):
    scaler = StandardScaler()
    expressions_scaled = scaler.fit_transform(expressions)
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(expressions_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', s=40, alpha=0.7, edgecolors='k', linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_grid(model, step_num, alpha):
    grid_data = np.zeros((model.grid.width, model.grid.height, 3))
    for agent in model.agents:
        x, y = agent.pos
        total = np.sum(agent.gene_expression) + 1e-10
        grid_data[x, y] = agent.gene_expression / total
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_data, origin='lower')
    ax.set_title(f'Клеточная сетка (α={alpha}, шаг {step_num})\nRGB = экспрессия генов', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ============================================================================
# ЗАПУСК ЭКСПЕРИМЕНТОВ
# ============================================================================
N, num_cells, num_steps = 20, 200, 100

for alpha_val in [0.0, 0.5]:
    print(f"\n{'='*60}\nЭКСПЕРИМЕНТ: α = {alpha_val}\n{'='*60}")
    
    model = CellDifferentiationModel(N=N, num_cells=num_cells, alpha=alpha_val, beta=0.1, sigma=0.05)
    for _ in range(num_steps):
        model.step()
    
    expressions = model.get_all_expressions()
    
    # 1. Оценка кластеризации (показывает 2 графика)
    optimal_k, best_sil = evaluate_clustering(expressions, max_k=8)
    print(f"→ Оптимальное число кластеров (по Silhouette): {optimal_k}")
    print(f"→ Лучший Silhouette Score: {best_sil:.3f}")
    
    # 2. Визуализация UMAP с найденным optimal_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(StandardScaler().fit_transform(expressions))
    visualize_umap(expressions, f'UMAP: α = {alpha_val} (Кластеров: {optimal_k})', labels)
    
    # 3. Визуализация сетки
    visualize_grid(model, num_steps, alpha_val)
