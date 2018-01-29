# Author: Marlos C. Machado

import random
import argparse
import numpy as np
import scipy as sp
import scipy.stats

from gridworld import GridWorld


class ArgsParser:
    """
    Read the user's input and parse the arguments properly. When returning args, each value is properly filled.
    Ideally one shouldn't have to read this function to access the proper arguments, but I postpone this.
    """

    @staticmethod
    def read_input_args():
        # Parse command line
        parser = argparse.ArgumentParser(
            description='Define algorithm\'s parameters.')

        parser.add_argument('-i', '--input', type=str, default='mdps/toy.mdp',
                            help='File containing the MDP definition (default: mdps/toy.mdp).')

        parser.add_argument('-o', '--output', type=str, default='graphs/',
                            help='Prefix that will be used to generate all outputs (default: graphs/).')

        parser.add_argument('-s', '--num_seeds', type=int, default=5,
                            help='Number of seeds to be averaged over when appropriate (default: 30).')

        parser.add_argument('-m', '--max_length_ep', type=int, default=100,
                            help='Maximum number of time steps an episode may last (default: 100).')

        parser.add_argument('-n', '--num_episodes', type=int, default=1000,
                            help='Number of episodes in which learning will happen (default: 1000).')

        args = parser.parse_args()

        return args


def mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def epsilon_greedy(q_values, epsilon=0.05):
    length_list = len(q_values)
    number = random.uniform(0, 1)
    if number < epsilon:
        return random.randrange(0, length_list)
    else:
        max_val = np.max(q_values)
        # I need to break ties randomly
        tmp_indx = []
        for i in range(length_list):
            if q_values[i] == max_val:
                tmp_indx.append(i)
        return random.choice(tmp_indx)


if __name__ == "__main__":
    # Read input arguments
    args = ArgsParser.read_input_args()

    # Create environment
    env = GridWorld(path=args.input)
    numStates = env.get_num_states()
    numActions = len(env.get_action_set())
    numRows, numCols = env.get_grid_dimensions()

    # Sarsa(0):
    gamma = 0.95
    step_size = 0.1
    num_steps_episode = []
    for seed in range(args.num_seeds):
        random.seed(seed)
        num_steps_episode.append([])
        q_values = np.zeros((numStates, numActions))
        for i in range(args.num_episodes):
            s = env.get_current_state()
            a = epsilon_greedy(q_values[s])
            num_steps = 0
            while num_steps < args.max_length_ep and not env.is_terminal():
                r = env.act(env.get_action_set()[a])
                next_s = env.get_current_state()
                next_a = epsilon_greedy(q_values[next_s])

                td_error = r + gamma * q_values[next_s][next_a] - q_values[s][a]
                q_values[s][a] = q_values[s][a] + step_size * td_error

                s = next_s
                a = next_a
                num_steps += 1
            env.reset()
            num_steps_episode[seed].append(num_steps)

    import matplotlib.pylab as plt

    mean, lower_bound, upper_bound = mean_confidence_interval(num_steps_episode)

    plt.plot(mean, color='red')
    plt.fill_between(range(args.num_episodes), lower_bound, upper_bound, color='red', alpha=0.4)
    plt.show()
