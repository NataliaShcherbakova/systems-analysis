# ============================================================
# ЗАДАНИЕ 4. Анализ коэкспрессии (Iris dataset)
# ============================================================

# 1. ИМПОРТЫ (Обязательно в начале каждого отдельного .py файла!)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# 2. Загрузка и предобработка данных
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X_log = np.log1p(X)  # логарифмирование для стабилизации дисперсии

# Переименуем для наглядности (моделируем «гены»)
gene_names = [
    'Ген A\n(sepal length)',
    'Ген B\n(sepal width)',
    'Ген C\n(petal length)',
    'Ген D\n(petal width)'
]

print(f"Данные: {X_log.shape[0]} образцов, {X_log.shape[1]} 'генов'")
print("\nОписательные статистики:")
print(X_log.describe().round(3))

# 3. Матрица взаимной информации
n_genes = X_log.shape[1]
mi_matrix = np.zeros((n_genes, n_genes))

for i in range(n_genes):
    for j in range(n_genes):
        if i != j:
            mi_matrix[i, j] = mutual_info_regression(
                X_log.iloc[:, i].values.reshape(-1, 1),
                X_log.iloc[:, j].values,
                random_state=42
            )[0]

# Визуализация матрицы MI
plt.figure(figsize=(8, 6))
sns.heatmap(mi_matrix, annot=True, fmt='.3f',
            xticklabels=gene_names, yticklabels=gene_names,
            cmap='YlOrRd', linewidths=0.5)
plt.title('Матрица взаимной информации между "генами"', fontsize=13)
plt.tight_layout()
plt.show()

# 4. Матрица корреляций Пирсона (для сравнения)
corr_matrix = X_log.corr(method='pearson').values

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f',
            xticklabels=gene_names, yticklabels=gene_names,
            cmap='coolwarm', center=0, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Матрица корреляций Пирсона', fontsize=13)
plt.tight_layout()
plt.show()

# 5. Сравнение MI и корреляции
print("\n=== Сравнение MI и |корреляции| ===")
print(f"{'Пара генов':<35} {'MI (бит)':<12} {'|Пирсон r|':<12}")
print("-" * 60)

for i in range(n_genes):
    for j in range(i+1, n_genes):
        mi = mi_matrix[i, j]
        corr = abs(corr_matrix[i, j])
        # Убираем переносы строк для красивого вывода в консоль
        name_i = gene_names[i].replace('\n', ' ')
        name_j = gene_names[j].replace('\n', ' ')
        print(f"{name_i:<18} vs {name_j:<15} {mi:<12.3f} {corr:<12.3f}")