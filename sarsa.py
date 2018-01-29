# Author: Marlos C. Machado

import utils
import random
import plotting
import numpy as np
from gridworld import GridWorld

if __name__ == "__main__":
    # Read input arguments
    args = utils.ArgsParser.read_input_args()

    # Create environment
    env = GridWorld(path=args.input)
    num_states = env.get_num_states()
    num_actions = len(env.get_action_set())
    num_rows, num_cols = env.get_grid_dimensions()

    # Sarsa(0):
    gamma = 0.95
    step_size = 0.1
    num_steps_episode = []
    for seed in range(args.num_seeds):
        random.seed(seed)
        num_steps_episode.append([])
        q_values = np.zeros((num_states, num_actions))
        for i in range(args.num_episodes):
            s = env.get_current_state()
            a = utils.epsilon_greedy(q_values[s])
            num_steps = 0
            while num_steps < args.max_length_ep and not env.is_terminal():
                r = env.act(env.get_action_set()[a])
                next_s = env.get_current_state()
                next_a = utils.epsilon_greedy(q_values[next_s])

                td_error = r + gamma * q_values[next_s][next_a] - q_values[s][a]
                q_values[s][a] = q_values[s][a] + step_size * td_error

                s = next_s
                a = next_a
                num_steps += 1
            env.reset()
            num_steps_episode[seed].append(num_steps)

        # Let me plot the max q-values for this seed:
        max_values = []
        for i in range(num_states):
            max_values.append(np.max(q_values[i]))
        plotting.plot_basis_function(args, num_rows, num_cols, np.array(max_values), 'max_q_seed_' + str(seed))

        # Let me plot the final policy for this seed:
        policy = []
        for i in range(num_states):
            policy.append(np.argmax(q_values[i]))
        plotting.plot_policy(env, args, num_rows, num_cols, policy, 'policy_seed_' + str(seed))

    # Finally, I'll just plot the results to provide examples on how to call the functions you might be interested at:
    plotting.plot_learning_curve(num_steps_episode, args.output)