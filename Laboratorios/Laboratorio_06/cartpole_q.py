# Importa las librerías necesarias
import gymnasium as gym  # Librería para el entorno de simulación del CartPole
import numpy as np       # Librería para operaciones numéricas y manipulación de arreglos
import matplotlib.pyplot as plt  # Librería para graficar resultados
import pickle           # Librería para serializar y guardar objetos en un archivo

# Función principal de entrenamiento y simulación
def run(is_training=True, render=False, max_episodes=1000):  
    # Crea el entorno CartPole, habilita la visualización si render es True
    env = gym.make('CartPole-v1', render_mode='human')

    # Define los límites de espacio para cada variable de estado del entorno
    pos_space = np.linspace(-2.4, 2.4, 10)  # Espacio para la posición del carrito
    vel_space = np.linspace(-4.8, 4.8 , 10)      # Espacio para la velocidad del carrito
    ang_space = np.linspace(-0.2095, 0.2095, 10)  # Espacio para el ángulo de la vara
    ang_vel_space = np.linspace(-4, 4, 10)  # Espacio para la velocidad angular de la vara
    #Tenemos 14641 estados posibles y dos acciones posibles
    # Inicializa la tabla Q con ceros, con dimensiones según el espacio de discretización y las acciones posibles
    q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))

    #env.action_space.n: número de acciones posibles en el entorno.

    # Define los parámetros de entrenamiento
    learning_rate_a = 0.1              # Tasa de aprendizaje alfa 
    discount_factor_g = 0.99           # Factor de descuento para el futuro valor de beta
    epsilon = 1                        # Valor inicial de epsilon (para exploración aleatoria)
    epsilon_decay_rate = 0.00001       # Tasa de decaimiento de epsilon
    rng = np.random.default_rng()      # Generador de números aleatorios para exploración

    rewards_per_episode = []           # Lista para almacenar recompensas de cada episodio

    # Bucle principal para cada episodio
    for i in range(max_episodes):
        # Determina si el episodio debe renderizarse
        current_render = render or (is_training and i % 5000 == 0)
        if current_render:
            env = gym.make('CartPole-v1', render_mode='human')
        else:
            env = gym.make('CartPole-v1')
        # Inicializa el estado y lo discretiza
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)  # Índice de la posición del carrito
        state_v = np.digitize(state[1], vel_space)  # Índice de la velocidad del carrito
        state_a = np.digitize(state[2], ang_space)  # Índice del ángulo de la vara
        state_av = np.digitize(state[3], ang_vel_space)  # Índice de la velocidad angular

        terminated = False    # Marca de finalización del episodio
        rewards = 0           # Acumulador de recompensa en el episodio

        # Bucle de interacción con el entorno hasta que termine el episodio
        while not terminated:
            # Selección de acción: exploración o explotación markov
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Exploración: elige acción aleatoria
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])  # Explotación: elige la mejor acción según Q

            # Ejecuta la acción en el entorno y observa el nuevo estado y recompensa
            # step nos devuelve el nuevo estado, la recompensa, si el episodio ha terminado
            new_state, reward, terminated, _, _ = env.step(action)
            # Discretiza el nuevo estado
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)
            # Actualización de la tabla Q si está en modo entrenamiento
            #eligen la probabilidad mas 
            if is_training:
                # ecuacion del bellman
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            # Actualiza el estado actual y acumula la recompensa
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
            rewards += reward

        # Guarda la recompensa total del episodio
        rewards_per_episode.append(rewards)
        # Calcula la media de recompensas de los últimos 100 episodios
        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode)-100):])

        # Imprime información de progreso cada 100 episodios durante el entrenamiento

        if is_training and i % 100 == 0:
            print(f'episodio: {i}  recompensa: {rewards}  Epsilon: {epsilon:0.2f}  recompensa media: {mean_rewards:0.2f}')

        # Criterio de parada: termina el entrenamiento si la recompensa media supera un umbral
        #if mean_rewards > 1000:
         #   print("Training complete: Mean rewards exceeded threshold.")
          #  break

        # Decaimiento de epsilon para disminuir exploración con el tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        env.close()  # Cierra el entorno para liberar recursos

    # Guarda la tabla Q en un archivo si está en modo entrenamiento
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Calcula y grafica la media de recompensas para observar la evolución del entrenamiento
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.plot(mean_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward (last 100 episodes)')
    plt.title('Training Progress')
    plt.savefig('cartpole0.png')  # Guarda la gráfica como imagen
    plt.show()  # Muestra la gráfica después de guardarla

# Ejecuta la función principal de entrenamiento y simulación
if __name__ == '__main__':
    run(is_training=True, render=False, max_episodes=100000)  # Entrenamiento
