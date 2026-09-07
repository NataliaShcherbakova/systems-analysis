import numpy as np
import matplotlib.pyplot as plt

# Простой пример: синтез и деградация белка
# Параметры
k_synthesis = 0.1  # скорость синтеза
k_degradation = 0.05  # скорость деградации
t_max = 100
dt = 0.1

# Детерминированное решение (ОДУ)
t = np.arange(0, t_max, dt)
P_deterministic = (k_synthesis / k_degradation) * (1 - np.exp(-k_degradation * t))

# Стохастическая траектория (упрощённый Гиллеспи)
np.random.seed(42)
P_stochastic = [0]
time = [0]
P = 0
t_current = 0

while t_current < t_max:
    # Вероятности реакций
    rate_synth = k_synthesis
    rate_degrad = k_degradation * P
    total_rate = rate_synth + rate_degrad

    if total_rate == 0:
        break

    # Время до следующей реакции
    tau = np.random.exponential(1 / total_rate)
    t_current += tau

    if t_current > t_max:
        break

    # Какая реакция произошла
    if np.random.random() < rate_synth / total_rate:
        P += 1  # синтез
    else:
        P = max(0, P - 1)  # деградация

    time.append(t_current)
    P_stochastic.append(P)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(t, P_deterministic, 'r--', linewidth=2, label='Детерминированная модель (ОДУ)')
plt.step(time, P_stochastic, 'b-', linewidth=1.5, label='Стохастическая модель (Гиллеспи)')
plt.xlabel('Время (сек)', fontsize=12)
plt.ylabel('Количество молекул белка', fontsize=12)
plt.title('Сравнение детерминированной и стохастической динамики', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gillespie_trajectory.png', dpi=300)
plt.show()