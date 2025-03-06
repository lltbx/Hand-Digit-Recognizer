import numpy as np
import matplotlib.pyplot as plt

# Параметри
r = 1  # Амплітуда (можна змінити на потрібне значення)
f = 10  # Частота 10 Гц (оскільки 20π = 2πf, f = 10)
T = 1 / f  # Період = 0.1 с
t = np.linspace(0, 0.3, 1000)  # Час від 0 до 0.3 с (3 періоди), 1000 точок

# Сигнал
x = r * np.cos(20 * np.pi * t)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(t, x, label=r'$x(t) = r \cos(20\pi t)$', color='blue')
plt.grid(True)
plt.xlabel('Час (с)')
plt.ylabel('Амплітуда')
plt.title('Графік сигналу \(x(t)\)')
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()