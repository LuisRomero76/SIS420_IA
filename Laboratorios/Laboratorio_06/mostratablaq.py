import pickle
import numpy as np

# Cargar el archivo .pkl
with open('./Laboratorios/Laboratorio_06/cartpole.pkl', 'rb') as f:
    data = pickle.load(f)

# # Verificar la estructura de los datos cargados
# print("Estructura de los datos cargados:")
# print(type(data))  # Imprime el tipo de datos (debería ser un ndarray o lista)

# # Si es un arreglo de NumPy (como la tabla Q), mostrar sus dimensiones
# if isinstance(data, np.ndarray):
#     print("\nDimensiones del arreglo Q:", data.shape)

# Mostrar los primeros estados y las correspondientes acciones en la tabla Q
# Asumimos que 'data' es una tabla Q de 4 dimensiones (estado + acción)
print("\nMuestra de los primeros estados y sus valores Q (acción 0 y 1):")
for i in range(min(10, data.shape[0])):  # Limitar a los primeros 5 estados
    for j in range(min(10, data.shape[1])):  # Limitar a las primeras 5 velocidades
        for k in range(min(10, data.shape[2])):  # Limitar a los primeros 5 ángulos
            for l in range(min(10, data.shape[3])):  # Limitar a las primeras 5 velocidades angulares
                q_values = data[i, j, k, l]  # Obtener los valores Q para un estado específico
                print(f"Estado ({i}, {j}, {k}, {l}) - Valores Q: {q_values}")
                print(f"Acción 0: {q_values[0]}, Acción 1: {q_values[1]}")  # Mostrar los valores de las dos acciones


