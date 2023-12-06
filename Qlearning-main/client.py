import connection as con

# Estabelecendo a conexão TCP
cn = con.connect(2037)

# Mesma ordem da Q-table
act = ["left", "right", "jump"]

while True:
	state, reward = con.get_state_reward(cn, act[2])
	print("Estado: ", state)
	print("Recompensa: ", reward)

	platform, direction = int(state[:7], 2), int(state[7:], 2)
	print("Plataforma: ", platform)
	print("Direção: ", direction)
