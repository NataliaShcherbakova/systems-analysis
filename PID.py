#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПИД-регулятор для биореактора
Практическое задание №11
Системный анализ в биоинформатике
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PIDConfig:
    """Конфигурация ПИД-регулятора"""
    Kp: float = 2.0  # Пропорциональный коэффициент
    Ki: float = 0.5  # Интегральный коэффициент
    Kd: float = 1.0  # Дифференциальный коэффициент
    dt: float = 0.1  # Шаг дискретизации (сек)
    u_max: float = 100.0  # Максимальная мощность (%)
    u_min: float = 0.0  # Минимальная мощность (%)


class BioreactorPID:
    """ПИД-регулятор для биореактора"""

    def __init__(self, config: PIDConfig):
        self.cfg = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.u_history = []

    def compute(self, setpoint: float, measurement: float) -> float:
        """Вычисление управляющего воздействия"""
        error = setpoint - measurement

        # Пропорциональная составляющая
        P = self.cfg.Kp * error

        # Интегральная составляющая (с anti-windup)
        self.integral += error * self.cfg.dt

        # Anti-windup: заморозка интеграла при насыщении
        if self.u_history:
            u_prev = self.u_history[-1]
            if u_prev >= self.cfg.u_max and error > 0:
                self.integral -= error * self.cfg.dt
            elif u_prev <= self.cfg.u_min and error < 0:
                self.integral -= error * self.cfg.dt

        I = self.cfg.Ki * self.integral

        # Дифференциальная составляющая
        derivative = (error - self.prev_error) / self.cfg.dt
        D = self.cfg.Kd * derivative

        # Суммарное управляющее воздействие
        u = P + I + D

        # Ограничение (saturation)
        u = np.clip(u, self.cfg.u_min, self.cfg.u_max)

        self.prev_error = error
        self.u_history.append(u)

        return u

    def reset(self):
        """Сброс регулятора"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.u_history = []


class BioreactorModel:
    """Математическая модель биореактора"""

    def __init__(self, T: float = 5.0, K: float = 2.0, dt: float = 0.1):
        self.T = T  # Постоянная времени
        self.K = K  # Коэффициент передачи
        self.dt = dt
        self.temperature = 20.0  # Начальная температура

    def step(self, u: float, disturbance: float = 0.0) -> float:
        """
        Один шаг моделирования
        u: управляющее воздействие (%)
        disturbance: внешнее возмущение (°C)
        """
        # Дифференциальное уравнение: T*dy/dt + y = K*u
        dy_dt = (self.K * u - self.temperature) / self.T
        self.temperature += dy_dt * self.dt + disturbance

        return self.temperature


def simulate_experiment(pid_config: PIDConfig,
                        setpoint: float = 37.0,
                        sim_time: float = 50.0,
                        disturbance_time: float = None,
                        disturbance_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Симуляция эксперимента

    Returns:
        times, temperatures, controls
    """
    pid = BioreactorPID(pid_config)
    reactor = BioreactorModel(dt=pid_config.dt)

    n_steps = int(sim_time / pid_config.dt)
    times = np.linspace(0, sim_time, n_steps)
    temperatures = []
    controls = []

    for t in times:
        # Внешнее возмущение (например, открытие двери)
        disturbance = 0.0
        if disturbance_time and abs(t - disturbance_time) < 1.0:
            disturbance = disturbance_value

        # Вычисление управления
        u = pid.compute(setpoint, reactor.temperature)

        # Шаг модели
        y = reactor.step(u, disturbance)

        temperatures.append(y)
        controls.append(u)

    return times, np.array(temperatures), np.array(controls)


def calculate_metrics(setpoint: float, temperatures: np.ndarray,
                      dt: float, settling_threshold: float = 0.05) -> dict:
    """
    Расчет метрик качества управления

    Returns:
        dict с метриками
    """
    error = np.abs(temperatures - setpoint)

    # Время нарастания (первое достижение setpoint)
    rise_idx = np.where(temperatures >= setpoint)[0]
    t_rise = rise_idx[0] * dt if len(rise_idx) > 0 else float('nan')

    # Перерегулирование
    max_temp = np.max(temperatures)
    overshoot = ((max_temp - setpoint) / setpoint) * 100 if max_temp > setpoint else 0.0

    # Время регулирования (вход в коридор ±5%)
    threshold = setpoint * settling_threshold
    in_band = np.abs(temperatures - setpoint) <= threshold

    # Находим последний выход из коридора
    exit_indices = np.where(~in_band)[0]
    if len(exit_indices) > 0:
        last_exit = exit_indices[-1]
        settling_idx = np.where(in_band[last_exit:])[0]
        t_settling = (last_exit + settling_idx[0]) * dt if len(settling_idx) > 0 else float('nan')
    else:
        settling_idx = np.where(in_band)[0]
        t_settling = settling_idx[0] * dt if len(settling_idx) > 0 else float('nan')

    # Статическая ошибка (последнее значение)
    steady_state_error = np.abs(temperatures[-1] - setpoint)

    # ISE (Integral Squared Error)
    ise = np.sum(error ** 2) * dt

    # IAE (Integral Absolute Error)
    iae = np.sum(error) * dt

    return {
        't_rise': round(t_rise, 2),
        'overshoot': round(overshoot, 2),
        't_settling': round(t_settling, 2),
        'steady_state_error': round(steady_state_error, 4),
        'ISE': round(ise, 4),
        'IAE': round(iae, 4)
    }


def plot_results(times: np.ndarray, temperatures: np.ndarray,
                 controls: np.ndarray, setpoint: float,
                 metrics: dict, experiment_name: str):
    """Визуализация результатов"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # График температуры
    ax1.plot(times, temperatures, 'b-', linewidth=2, label='Температура')
    ax1.axhline(y=setpoint, color='r', linestyle='--', label=f'Задание ({setpoint}°C)')
    ax1.axhline(y=setpoint * 1.05, color='g', linestyle=':', alpha=0.5)
    ax1.axhline(y=setpoint * 0.95, color='g', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Температура (°C)', fontsize=11)
    ax1.set_title(f'{experiment_name}\n' +
                  f'Перерегулирование: {metrics["overshoot"]}%, ' +
                  f'Время регулирования: {metrics["t_settling"]}с',
                  fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # График управления
    ax2.plot(times, controls, 'orange', linewidth=1.5)
    ax2.set_ylabel('Мощность нагревателя (%)', fontsize=11)
    ax2.set_xlabel('Время (сек)', fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{experiment_name.replace(" ", "_")}.png', dpi=150)
    plt.show()


def print_metrics_table(metrics: dict, experiment_name: str):
    """Вывод таблицы метрик"""
    print(f"\n{'=' * 60}")
    print(f"{experiment_name}")
    print(f"{'=' * 60}")
    print(f"{'Метрика':<30} {'Значение':>15}")
    print(f"{'-' * 60}")
    for key, value in metrics.items():
        metric_name = {
            't_rise': 'Время нарастания t_p (сек)',
            'overshoot': 'Перерегулирование σ% (%)',
            't_settling': 'Время регулирования t_рег (сек)',
            'steady_state_error': 'Статическая ошибка (°C)',
            'ISE': 'ISE (кв.°C·сек)',
            'IAE': 'IAE (°C·сек)'
        }.get(key, key)
        print(f"{metric_name:<30} {value:>15.4f}")
    print(f"{'=' * 60}\n")


# ============================================================================
# ЗАДАНИЕ ДЛЯ СТУДЕНТА: Изменяйте параметры в этом блоке
# ============================================================================

if __name__ == "__main__":
    print("ПИД-регулятор для биореактора")
    print("Практическое задание №11\n")

    # ========================================================================
    # ЭКСПЕРИМЕНТ 1: Только P-регулятор
    # ========================================================================
    print("\n🔬 ЭКСПЕРИМЕНТ 1: P-регулятор")
    config_1 = PIDConfig(
        Kp=2.0,  # ← МЕНЯЙТЕ это значение
        Ki=0.0,  # Интегральная отключена
        Kd=0.0,  # Дифференциальная отключена
        dt=0.1
    )

    times_1, temps_1, controls_1 = simulate_experiment(config_1, setpoint=37.0)
    metrics_1 = calculate_metrics(37.0, temps_1, config_1.dt)
    print_metrics_table(metrics_1, "Эксперимент 1: P-регулятор")

    # ========================================================================
    # ЭКСПЕРИМЕНТ 2: PI-регулятор
    # ========================================================================
    print("\n ЭКСПЕРИМЕНТ 2: PI-регулятор")
    config_2 = PIDConfig(
        Kp=2.0,  # ← МЕНЯЙТЕ это значение
        Ki=0.5,  # ← МЕНЯЙТЕ это значение
        Kd=0.0,  # Дифференциальная отключена
        dt=0.1
    )

    times_2, temps_2, controls_2 = simulate_experiment(config_2, setpoint=37.0)
    metrics_2 = calculate_metrics(37.0, temps_2, config_2.dt)
    print_metrics_table(metrics_2, "Эксперимент 2: PI-регулятор")

    # ========================================================================
    # ЭКСПЕРИМЕНТ 3: PID-регулятор
    # ========================================================================
    print("\n🔬 ЭКСПЕРИМЕНТ 3: PID-регулятор")
    config_3 = PIDConfig(
        Kp=2.0,  # ← МЕНЯЙТЕ это значение
        Ki=0.5,  # ← МЕНЯЙТЕ это значение
        Kd=1.0,  # ← МЕНЯЙТЕ это значение
        dt=0.1
    )

    times_3, temps_3, controls_3 = simulate_experiment(config_3, setpoint=37.0)
    metrics_3 = calculate_metrics(37.0, temps_3, config_3.dt)
    print_metrics_table(metrics_3, "Эксперимент 3: PID-регулятор")

    # ========================================================================
    # ЭКСПЕРИМЕНТ 4: "Агрессивный" PID (большие коэффициенты)
    # ========================================================================
    print("\n🔬 ЭКСПЕРИМЕНТ 4: Агрессивный PID")
    config_4 = PIDConfig(
        Kp=10.0,  # ← МЕНЯЙТЕ это значение (попробуйте 5, 10, 15)
        Ki=2.0,  # ← МЕНЯЙТЕ это значение
        Kd=0.5,  # ← МЕНЯЙТЕ это значение
        dt=0.1
    )

    times_4, temps_4, controls_4 = simulate_experiment(config_4, setpoint=37.0)
    metrics_4 = calculate_metrics(37.0, temps_4, config_4.dt)
    print_metrics_table(metrics_4, "Эксперимент 4: Агрессивный PID")

    # ========================================================================
    # Сводная таблица результатов
    # ========================================================================
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"{'Эксперимент':<15} {'Kp':>6} {'Ki':>6} {'Kd':>6} | "
          f"{'t_p (сек)':>10} {'σ%':>8} {'t_рег (сек)':>12} {'e_уст':>8}")
    print("-" * 80)

    experiments = [
        ("P-регулятор", config_1, metrics_1),
        ("PI-регулятор", config_2, metrics_2),
        ("PID-регулятор", config_3, metrics_3),
        ("Агрессивный", config_4, metrics_4)
    ]

    for name, cfg, met in experiments:
        print(f"{name:<15} {cfg.Kp:>6.1f} {cfg.Ki:>6.1f} {cfg.Kd:>6.1f} | "
              f"{met['t_rise']:>10.2f} {met['overshoot']:>7.2f} "
              f"{met['t_settling']:>12.2f} {met['steady_state_error']:>8.4f}")

    print("=" * 80)

    # Визуализация
    plot_results(times_3, temps_3, controls_3, 37.0, metrics_3,
                 "PID-регулятор (оптимальный)")

    print("\n✅ Графики сохранены в PNG-файлы")
    print("📊 Заполните таблицу в отчёте на основе полученных метрик")