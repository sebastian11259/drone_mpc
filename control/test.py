import pybullet as p



# X = [-2,0,0]
# Q = [0, 0, 0, 1]
# Q1 = [0.0,0.0,-0.7071068, 0.7071068]
# Q2 = [0, 0, -1, 0]
# Q3 = [0.0,0.0,0.7071068, 0.7071068]
#
# print(p.invertTransform(X, Q))
# print(p.invertTransform(X, Q1))
# print(p.invertTransform(X, Q2))
# print(p.invertTransform(X, Q3))

import numpy as np
import matplotlib.pyplot as plt


# Funkcja reprezentująca równanie różniczkowe dy/dt = -y
def derivative(y):
    return -y


# Metoda Eulera
def euler_method(y0, h, steps):
    t_values = [0]  # Czas zaczyna się od 0
    y_values = [y0]  # Początkowa wartość y

    for i in range(steps):
        t = t_values[-1]
        y = y_values[-1]
        y_next = y + h * derivative(y)

        t_values.append(t + h)
        y_values.append(y_next)

    return t_values, y_values


# Obliczenia dla dwóch różnych wartości kroku
h1 = 1
h2 = 0.1
steps1 = 10  # 10 kroków dla h = 1
steps2 = 100  # 3 kroki dla h = 3

t_values1, y_values1 = euler_method(1, h1, steps1)
t_values2, y_values2 = euler_method(1, h2, steps2)

# Wykresy
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_values1, y_values1, marker='o', label=f'steps = {steps1}, h = {h1}')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values2, y_values2, marker='o', label=f'steps = {steps2}, h = {h2}')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
