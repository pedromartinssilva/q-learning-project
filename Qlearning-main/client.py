import connection as con
import random

def epsilon_greedy(q_table, state, epsilon) -> int:
	"""
	Chooses an action based on epsilon-greedy policy.

	Parameters:
		q_table (list): Table representing the expected rewards for each action in each state.
        state (str): Current state in binary format.
        epsilon (float): Exploration rate.
	
	Returns:
	    int: Index of the selected action.
	"""
	state_index = int(state, 2)
	if random.random() < epsilon:
		# Exploration: random action
		return random.choice(range(len(q_table[state_index])))
	else:
		# Exploitation: best action based on the Q-table
		return max(range(len(q_table[state_index])), key=lambda a: q_table[state_index][a])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
	"""
	Updates the Q-table based on the Q-learning algorithm.

	Parameters:
		q_table (list): Q-table representing the expected rewards for each action in each state.
		state (str): Current state in binary format.
		action (int): Index of the selected action.
		reward (float): Reward obtained from taking the action.
		next_state (str): Next state in binary format.
		alpha (float): Learning rate.
		gamma (float): Discount factor.
	"""
	state_index = int(state, 2)
	next_state_index = int(next_state, 2)
	q_predict = q_table[state_index][action]
	q_target = reward + gamma * max(q_table[next_state_index])
	q_table[state_index][action] += alpha * (q_target - q_predict)

def main():
	"""
	The Q-table is updated iteratively based on the rewards obtained from the actions taken.
	The final Q-table is saved to 'resultado.txt'.
	"""

	# Establishing TCP connection
	cn = con.connect(2037)

	# Defining possible actions
	actions = ["left", "right", "jump"]

	# Initializing Q-table
	file_path = "resultado.txt"
	with open(file_path, 'r') as file:
		q_table = [list(map(float, line.split())) for line in file]
	
	# Setting parameters
	num_episodes = 1
	num_steps    = 1000

	epsilon = 0.1	# Exploration rate
	alpha = 0.1		# Learning rate or step size
	gamma = 0.9		# Discount factor 

    # Main loop
	for episode in range(num_episodes):
		# Getting initial state and reward
		terminal = 0
		action = "jump"
		state, reward = con.get_state_reward(cn, action)
		total_reward = reward

		for _ in range(num_steps):
			# Selecting action based on epsilon-greedy policy
			action_index = epsilon_greedy(q_table, state, epsilon)
			action = actions[action_index]

			# Executing the action and getting the next state and reward
			next_state, reward = con.get_state_reward(cn, action)
			if reward == 1000: terminal+=1

			# Updating the Q-table
			update_q_table(q_table, state, action_index, reward, next_state, alpha, gamma)

			state = next_state
			total_reward += reward

		print(f"In {num_steps} steps, amongois got to the terminal {terminal} times!")
    # Saving Q-table to resultado.txt 
	# with open(file_path, "w") as file:
	# 	for row in q_table:
	# 		file.write(" ".join(f"{value:.6f}" for value in row) + "\n")

if __name__ == "__main__":
    main()