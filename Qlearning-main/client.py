import connection as con
import numpy as np

def choose_action(q_table, state, epsilon):
	if np.random.rand() < epsilon:
		# Exploração: ação aleatória
		return np.random.choice(len(q_table[state]))
	else:
		# Exploração: melhor ação com base na Q-table
		state_index = int(state, 2)
		return np.argmax(q_table[state_index, :])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
	state_index = int(state, 2)
	next_state_index = int(next_state, 2)
	q_predict = q_table[state_index, action]
	q_target = reward + gamma * np.max(q_table[next_state_index])
	q_table[state_index, action] += alpha * (q_target - q_predict)

def main():
	# Estabelecendo a conexão TCP
	cn = con.connect(2037)

	# Definindo ações possíveis
	actions = ["left", "right", "jump"]
	num_actions = len(actions)
	num_states = 96

	# Inicializando Q-table
	q_table = np.zero((num_states, num_actions))

	# Estabelecendo parâmetros
	num_platforms 	= 3
	num_directions 	= 4
	num_states 		= num_platforms * num_directions
	num_iter		= 100

	epsilon = 0.1 # Taxa de exploração
	alpha	= 0.1 # Taxa de aprendizado
	gamma	= 0.9 # Fator de desconto		

	# Loop principal
	for iter in range(num_iter):
		# Obtendo estado e recompensa
		state = con.get_initial_state(cn)
		total_reward = 0
		print("Estado inicial: ", state)

		while not con.is_terminal_state(state):
			# Selecionando ação com base na política epsilon-greedy
			action_index = choose_action(q_table, state, epsilon)
			action = actions[action_index]

			# Executando a ação e obtendo o próximo estado e recompensa
			next_state, reward = con.get_state_reward(cn, action)

			# Atualizando a Q-table
			update_q_table(q_table, state, action, reward, next_state, alpha, gamma)

			state = next_state
			total_reward += reward

		# Extraindo informações do estado
		#platform, direction = int(state[:7], 2), int(state[7:], 2)
		# print("Plataforma: ", platform)
		# print("Direção: ", direction)

		print(f"Iteração {num_iter + 1}/{num_iter}, Recompensa: {total_reward}")

	# Salvando Q-table no arquivo 'resultado.txt'
	np.savetxt("resultado.txt", q_table, fmt="%.6f")

if __name__ == "__main__":
	main()