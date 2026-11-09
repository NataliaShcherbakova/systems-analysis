#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПИД-регулятор для биореактора
Практическое задание №11
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

# ============================================================================
# БЛОК НАСТРОЕК
# ============================================================================
# Инструкция по выполнению:
# 1. Эксперимент 1 (P): Меняйте Kp (например, 1.0, 2.0, 5.0). Смотрите на ошибку.
# 2. Эксперимент 2 (PI): Меняйте Ki при Kp=1.5. Смотрите, как исчезает ошибка.
# 3. Эксперимент 3 (PID): Меняйте Kd при Kp=2.5, Ki=1.5. Смотрите на гашение колебаний.
# 4. Эксперимент 4 (Агрессивный): Оставьте эти параметры, чтобы увидеть "раскачку" (колебания).

CONFIG_1 = dict(Kp=1.0, Ki=0.0, Kd=0.0)  # P-регулятор
CONFIG_2 = dict(Kp=1.5, Ki=0.8, Kd=0.0)  # PI-регулятор
CONFIG_3 = dict(Kp=2.5, Ki=1.5, Kd=1.5)  # PID-регулятор
CONFIG_4 = dict(Kp=3.0, Ki=8.0, Kd=0.0)  # Агрессивный PID (с колебаниями)

SIM_TIME = 15.0  # Время симуляции (сек)
SETPOINT = 37.0  # Задание температуры
DT = 0.1  # Шаг дискретизации


# ============================================================================


@dataclass
class PIDConfig:
    Kp: float = 2.0
    Ki: float = 0.5
    Kd: float = 1.0
    dt: float = 0.1
    u_max: float = 100.0
    u_min: float = 0.0


class BioreactorPID:
    def __init__(self, config: PIDConfig):
        self.cfg = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.u_history = []

    def compute(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement
        P = self.cfg.Kp * error

        self.integral += error * self.cfg.dt
        if self.u_history:
            u_prev = self.u_history[-1]
            if u_prev >= self.cfg.u_max and error > 0:
                self.integral -= error * self.cfg.dt
            elif u_prev <= self.cfg.u_min and error < 0:
                self.integral -= error * self.cfg.dt
        I = self.cfg.Ki * self.integral

        derivative = (error - self.prev_error) / self.cfg.dt
        D = self.cfg.Kd * derivative

        u = P + I + D
        u = np.clip(u, self.cfg.u_min, self.cfg.u_max)

        self.prev_error = error
        self.u_history.append(u)
        return u


class BioreactorModel:
    def __init__(self, T: float = 5.0, K: float = 2.0, dt: float = 0.1):
        self.T = T
        self.K = K
        self.dt = dt
        self.temperature = 20.0

    def step(self, u: float, disturbance: float = 0.0) -> float:
        dy_dt = (self.K * u - self.temperature) / self.T
        self.temperature += dy_dt * self.dt + disturbance
        return self.temperature


def simulate_experiment(pid_config: PIDConfig, setpoint: float, sim_time: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    pid = BioreactorPID(pid_config)
    reactor = BioreactorModel(dt=pid_config.dt)
    n_steps = int(sim_time / pid_config.dt)
    times = np.linspace(0, sim_time, n_steps)
    temperatures, controls = [], []

    for _ in times:
        u = pid.compute(setpoint, reactor.temperature)
        y = reactor.step(u, 0.0)
        temperatures.append(y)
        controls.append(u)

    return times, np.array(temperatures), np.array(controls)


def calculate_metrics(setpoint: float, temperatures: np.ndarray, dt: float) -> dict:
    error = np.abs(temperatures - setpoint)
    rise_idx = np.where(temperatures >= setpoint)[0]
    t_rise = rise_idx[0] * dt if len(rise_idx) > 0 else float('nan')

    max_temp = np.max(temperatures)
    overshoot = ((max_temp - setpoint) / setpoint) * 100 if max_temp > setpoint else 0.0

    in_band = np.abs(temperatures - setpoint) <= (setpoint * 0.05)
    exit_indices = np.where(~in_band)[0]
    if len(exit_indices) > 0:
        last_exit = exit_indices[-1]
        settling_idx = np.where(in_band[last_exit:])[0]
        t_settling = (last_exit + settling_idx[0]) * dt if len(settling_idx) > 0 else float('nan')
    else:
        settling_idx = np.where(in_band)[0]
        t_settling = settling_idx[0] * dt if len(settling_idx) > 0 else float('nan')

    steady_state_error = np.abs(temperatures[-1] - setpoint)
    ise = np.sum(error ** 2) * dt
    iae = np.sum(error) * dt

    return {
        't_rise': round(t_rise, 2), 'overshoot': round(overshoot, 2),
        't_settling': round(t_settling, 2), 'steady_state_error': round(steady_state_error, 4),
        'ISE': round(ise, 4), 'IAE': round(iae, 4)
    }


def plot_all_experiments(experiments_data: list):
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    fig, axes = plt.subplots(len(experiments_data), 2, figsize=(11, 7.5))
    if len(experiments_data) == 1: axes = np.array([axes])

    for i, (name, times, temps, controls, metrics) in enumerate(experiments_data):
        ax_temp = axes[i, 0]
        ax_temp.plot(times, temps, 'b-', linewidth=1.5, label='Температура')
        ax_temp.axhline(y=SETPOINT, color='r', linestyle='--', linewidth=1, label=f'Задание ({SETPOINT} C)')
        ax_temp.axhline(y=SETPOINT * 1.05, color='g', linestyle=':', alpha=0.5, linewidth=1)
        ax_temp.axhline(y=SETPOINT * 0.95, color='g', linestyle=':', alpha=0.5, linewidth=1)
        ax_temp.set_ylabel('Температура (C)')
        ax_temp.set_title(f'{name}\nσ = {metrics["overshoot"]}%, t_рег = {metrics["t_settling"]} c')
        ax_temp.grid(True, alpha=0.3)
        if i == 0: ax_temp.legend(loc='lower right', fontsize=8)

        ax_ctrl = axes[i, 1]
        ax_ctrl.plot(times, controls, 'orange', linewidth=1.2)
        ax_ctrl.set_ylabel('Мощность (%)')
        ax_ctrl.set_ylim(-5, 105)
        ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.set_title(f'Управление\nISE = {metrics["ISE"]}, IAE = {metrics["IAE"]}')

    axes[-1, 0].set_xlabel('Время (сек)')
    axes[-1, 1].set_xlabel('Время (сек)')

    plt.tight_layout()
    plt.savefig("PID.png", dpi=150)  # Сохранение в файл PID.png
    plt.show()


if __name__ == "__main__":
    print("ПИД-регулятор для биореактора")
    print("Практическое задание №11\n")

    exp_configs = [
        ("Эксперимент 1: P-регулятор", CONFIG_1),
        ("Эксперимент 2: PI-регулятор", CONFIG_2),
        ("Эксперимент 3: PID-регулятор", CONFIG_3),
        ("Эксперимент 4: Агрессивный PID", CONFIG_4)
    ]

    results = []

    print(f"{'Эксперимент':<20} {'Kp':>5} {'Ki':>5} {'Kd':>5} | "
          f"{'t_p':>6} {'σ%':>6} {'t_рег':>6} {'e_уст':>8} {'ISE':>8} {'IAE':>8}")
    print("-" * 85)

    for name, cfg_dict in exp_configs:
        config = PIDConfig(Kp=cfg_dict['Kp'], Ki=cfg_dict['Ki'], Kd=cfg_dict['Kd'], dt=DT)
        times, temps, controls = simulate_experiment(config, SETPOINT, SIM_TIME)
        metrics = calculate_metrics(SETPOINT, temps, DT)

        results.append((name.split(": ")[1], times, temps, controls, metrics))

        # Вывод таблицы в консоль
        print(f"{name:<20} {config.Kp:>5.1f} {config.Ki:>5.1f} {config.Kd:>5.1f} | "
              f"{metrics['t_rise']:>6.2f} {metrics['overshoot']:>6.2f} "
              f"{metrics['t_settling']:>6.2f} {metrics['steady_state_error']:>8.4f} "
              f"{metrics['ISE']:>8.2f} {metrics['IAE']:>8.2f}")

    print("-" * 85)

    plot_all_experiments(results)
    print("\n✅ График сохранен в файл: PID.png")
