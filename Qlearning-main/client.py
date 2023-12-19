import connection as con
import random

def epsilon_greedy(q_table, state, epsilon) -> int:
	"""
	Chooses an action based on epsilon-greedy policy.

	Parameters:
		q_table (list): Q-table representing the expected rewards for each action in each state.
        state (str): Current state in binary format.
        epsilon (float): Exploration rate.
	
	Returns:
	    int: Index of the selected action.
	"""
	if random.random() < epsilon:
		# Exploration: random action
		return random.choice(range(len(q_table[state])))
	else:
		# Exploitation: best action based on the Q-table
		state_index = int(state, 2)
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
	The Q-table is updated iteratively based on the rewards obtained from taking actions in the environment.
	The final Q-table is saved to a file named 'resultado.txt'.
	"""

	# Establishing TCP connection
	cn = con.connect(2037)

	# Defining possible actions
	actions = ["left", "right", "jump"]
	num_actions = len(actions)
	num_states = 96

	# Initializing Q-table
	q_table = [[0.0] * num_actions for _ in range(num_states)]

	# Setting parameters
	num_platforms = 3
	num_directions = 4
	num_states = num_platforms * num_directions
	num_iter = 100

	"""
	The epsilon parameter introduces randomness into the algorithm, forcing different actions.
	If epsilon is set to 0, we never explore but always exploit the knowledge we already have.
	Having the epsilon set to 1 force the algorithm to always take random actions and never use past knowledge.
	Usually, epsilon is selected as a small number close to 0.
	"""
	epsilon = 0.1  	# Exploration rate
	"""
	If we set alpha to zero, the agent learns nothing from new actions.
	If we set alpha to 1, the agent completely ignores prior knowledge and only values the most recent information.
	Higher alpha values make Q-values change faster.
	"""
	alpha = 0.1		# Learning rate or step size
	"""
	If we set gamma to zero, the agent completely ignores the future rewards. Such agents only consider current rewards.
	If we set gamma to 1, the algorithm would look for high rewards in the long term.
	A high gamma value might prevent conversion: summing up non-discounted rewards leads to having high Q-values.
	"""
	gamma = 0.9    	# Discount factor 

    # Main loop
	for iter in range(num_iter):
		# Getting initial state and reward
		action = random.choice(actions)
		state, reward = con.get_state_reward(cn, action)
		total_reward = reward
		print("Initial state: ", state)

		while not con.is_terminal_state(state):
			# Selecting action based on epsilon-greedy policy
			action_index = epsilon_greedy(q_table, state, epsilon)
			action = actions[action_index]

			# Executing the action and getting the next state and reward
			next_state, reward = con.get_state_reward(cn, action)

			# Updating the Q-table
			update_q_table(q_table, state, action_index, reward, next_state, alpha, gamma)

			state = next_state
			total_reward += reward

		print(f"Iteration {iter + 1}/{num_iter}, Total Reward: {total_reward}")

    # Saving Q-table to the 'resultado.txt' file
	with open("resultado.txt", "w") as file:
		for row in q_table:
			file.write(" ".join(f"{value:.6f}" for value in row) + "\n")

if __name__ == "__main__":
    main()