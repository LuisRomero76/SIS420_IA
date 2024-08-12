import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(40)

estaturas = np.random.uniform(1.4, 2.0, 100)

pesos = []

for estatura in estaturas:
    
    peso_min = 18.5 * (estatura ** 2)  # Peso mínimo según IMC de 18.5
    peso_max = 24.9 * (estatura ** 2)  # Peso máximo según IMC de 24.9
    
    # Generar un peso aleatorio entre el peso mínimo y máximo calculado
    peso = np.random.uniform(peso_min, peso_max)
    
    pesos.append(peso)  # Añadir el peso a la lista de pesos

data = pd.DataFrame({
    'Estatura (m)': estaturas,
    'Peso (kg)': pesos
})

tabla = tabulate(data.head(100), headers='keys', tablefmt="grid")
print(tabla)

x = data['Estatura (m)']
y = data['Peso (kg)']
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
b = np.mean(y) - m * np.mean(x)

interseccion = m * x + b

plt.scatter(data['Estatura (m)'], data['Peso (kg)'], color='darkblue', label='Datos Encontrados')
plt.plot(x, interseccion, color='red', label='Línea ajustada')
plt.title('Estatura vs Peso con Línea Ajustada')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()
