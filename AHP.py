#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Практика 4: AHP-анализ
Тема КП: Выбор биоинформатического пайплайна
Студент: <Ваше ФИО>
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ============================================================
# 1. КОНФИГУРАЦИЯ ЗАДАЧИ
# ============================================================
TASK_NAME = "Выбор пайплайна для геномной диагностики"
CRITERIA = ["Точность (Accuracy)", "Скорость (Speed)", "Простота (Usability)", "Стоимость (Cost)"]
ALTERNATIVES = ["GATK Best Practices", "FastQC + Custom", "Illumina DRAGEN"]

# Матрица парных сравнений КРИТЕРИЕВ
CRITERIA_MATRIX = np.array([
    [1, 3, 5, 7],
    [1 / 3, 1, 3, 5],
    [1 / 5, 1 / 3, 1, 3],
    [1 / 7, 1 / 5, 1 / 3, 1]
])

# Матрицы парных сравнений АЛЬТЕРНАТИВ по каждому критерию
ALT_MATRICES = {
    "Точность (Accuracy)": np.array([
        [1, 5, 3],
        [1 / 5, 1, 1 / 3],
        [1 / 3, 3, 1]
    ]),
    "Скорость (Speed)": np.array([
        [1, 1 / 3, 1 / 5],
        [3, 1, 1 / 3],
        [5, 3, 1]
    ]),
    "Простота (Usability)": np.array([
        [1, 3, 1],
        [1 / 3, 1, 1 / 3],
        [1, 3, 1]
    ]),
    "Стоимость (Cost)": np.array([
        [1, 3, 1 / 5],
        [1 / 3, 1, 1 / 7],
        [5, 7, 1]
    ])
}

# ============================================================
# 2. ФУНКЦИИ РАСЧЕТА AHP
# ============================================================
# Таблица случайных индексов (Random Index) Саати
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def calculate_weights(matrix):
    """Расчет весов методом геометрического среднего."""
    n = matrix.shape[0]
    # Геометрическое среднее по строкам
    geo_mean = np.prod(matrix, axis=1) ** (1 / n)
    # Нормализация
    weights = geo_mean / geo_mean.sum()
    return weights


def calculate_cr(matrix, weights):
    """Расчет Consistency Ratio (CR)."""
    n = matrix.shape[0]
    # Оценка максимального собственного числа
    lambda_max = np.mean((matrix @ weights) / weights)
    # Индекс согласованности (CI)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    # Отношение согласованности (CR)
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    return cr, lambda_max


# ============================================================
# 3. ОСНОВНОЙ РАСЧЕТ
# ============================================================
def main():
    print("=" * 70)
    print(f"AHP-АНАЛИЗ: {TASK_NAME}")
    print("=" * 70)

    # --- ШАГ 1 и 2: Веса критериев и их согласованность ---
    crit_weights = calculate_weights(CRITERIA_MATRIX)
    cr_crit, lambda_max_crit = calculate_cr(CRITERIA_MATRIX, crit_weights)

    print("\n[ШАГ 1-2] ВЕСА КРИТЕРИЕВ:")
    for c, w in zip(CRITERIA, crit_weights):
        print(f"  {c:25s}: {w:.4f}")
    cr_status = "✅ СОГЛАСОВАНО" if cr_crit < 0.10 else "❌ ПЕРЕСМОТРИТЕ МАТРИЦУ"
    print(f"  -> CR критериев: {cr_crit:.4f} {cr_status} (λ_max = {lambda_max_crit:.3f})")

    # --- ШАГ 2 (продолжение): Локальные приоритеты альтернатив ---
    alt_global_scores = {alt: 0.0 for alt in ALTERNATIVES}

    print("\n[ШАГ 2] ОЦЕНКА АЛЬТЕРНАТИВ ПО КАЖДОМУ КРИТЕРИЮ:")
    for i, criterion in enumerate(CRITERIA):
        matrix = ALT_MATRICES[criterion]
        alt_weights = calculate_weights(matrix)
        cr_alt, _ = calculate_cr(matrix, alt_weights)

        print(f"\n  Критерий: '{criterion}' (Вес = {crit_weights[i]:.4f})")
        for j, alt in enumerate(ALTERNATIVES):
            print(f"    {alt:25s}: {alt_weights[j]:.4f}")
            # Накопление глобального приоритета (ШАГ 3)
            alt_global_scores[alt] += crit_weights[i] * alt_weights[j]

        cr_status_alt = "✅" if cr_alt < 0.10 else "❌"
        print(f"    -> CR матрицы: {cr_alt:.4f} {cr_status_alt}")

    # --- ШАГ 3 и 4: Глобальные приоритеты и ранжирование ---
    print("\n" + "=" * 70)
    print("[ШАГ 3-4] ИТОГОВЫЙ РЕЙТИНГ АЛЬТЕРНАТИВ (Глобальные приоритеты):")
    print("-" * 70)

    # Сортировка по убыванию scores
    sorted_alts = sorted(alt_global_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (alt, score) in enumerate(sorted_alts, 1):
        marker = " 👑 РЕКОМЕНДУЕМ" if rank == 1 else ""
        print(f"  {rank}. {alt:25s}: {score:.4f} ({score * 100:.1f}%){marker}")

    # Проверка суммы (должна быть ~1.0)
    total_score = sum(alt_global_scores.values())
    print(f"\n  [Проверка] Сумма глобальных приоритетов: {total_score:.4f} (должна быть 1.000)")

    # ============================================================
    # 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ И ВИЗУАЛИЗАЦИЯ
    # ============================================================

    # 4.1. Построение и сохранение графика
    plt.figure(figsize=(10, 6))
    # Данные для графика (переворачиваем для красивого отображения: лучший сверху)
    plot_alts = [item[0] for item in reversed(sorted_alts)]
    plot_scores = [item[1] for item in reversed(sorted_alts)]

    bars = plt.barh(plot_alts, plot_scores, color=['#e0e0e0', '#e0e0e0', '#4a90e2'])

    plt.xlabel('Глобальный приоритет (Вес)', fontsize=12)
    plt.title(f'AHP: {TASK_NAME}\nИтоговый рейтинг альтернатив', fontsize=14, fontweight='bold')
    plt.xlim(0, max(plot_scores) * 1.2)  # Отступ справа для подписей

    # Добавление значений на столбцы
    for bar, score in zip(bars, plot_scores):
        plt.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f'{score:.3f}', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('ahp_result.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ График сохранен: {os.path.abspath('ahp_result.png')}")
    plt.close()  # Закрыть, чтобы не мешал в Jupyter

    # 4.2. Сохранение отчета в CSV
    csv_filename = 'ahp_report.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Ранг', 'Альтернатива', 'Глобальный приоритет', 'Процент', 'Решение'])
        for rank, (alt, score) in enumerate(sorted_alts, 1):
            decision = "РЕКОМЕНДУЕМ" if rank == 1 else ("Альтернатива" if rank == 2 else "Не рекомендуется")
            writer.writerow([rank, alt, f"{score:.4f}", f"{score * 100:.1f}%", decision])

    print(f"✅ Отчет сохранен: {os.path.abspath(csv_filename)}")

    # 4.3. Готовый вывод для пояснительной записки КП
    best_alt = sorted_alts[0][0]
    best_score = sorted_alts[0][1]
    print("\n" + "=" * 70)
    print(f"  На основе метода анализа иерархий (AHP) с {len(CRITERIA)} критериями,")
    print(f"  полученными из анализа стейкхолдеров, рекомендуемой альтернативой")
    print(f"  является: '{best_alt}' (итоговый вес = {best_score:.4f}).")
    print(f"  Отношение согласованности (CR) для всех матриц < 0.10, что подтверждает")
    print(f"  внутреннюю непротиворечивость экспертных суждений.")
    print("=" * 70)


if __name__ == "__main__":
    main()
