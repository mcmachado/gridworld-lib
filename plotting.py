# Author Marlos C. Machado

import utils
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
# I need this for the 3d projection:
from mpl_toolkits.mplot3d import Axes3D


def plot_basis_function(args, x_range, y_range, basis, prefix):
    """
    Plots 3d graph where the x and y coordinates represent the grid and the z axis encodes the value to be depicted.
    One could use this to look at q-values, for example. In my work, I've used this plot to show proto-value functions,
    the successor representation, or eigenvectors.

    :param args: list of all arguments passed at the execution so we can access the paths we are interested at
    :param x_range: one of the dimensions of the grid world
    :param y_range: the other dimension of the grid world
    :param basis: actual values to be plotted
    :param prefix: string that will be used as the filename of the generated graph
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    data_x, data_y = np.meshgrid(np.arange(y_range), np.arange(x_range))
    data_z = basis.reshape(x_range, y_range)

    for ii in range(len(data_x)):
        for j in range(int(len(data_x[ii]) / 2)):
            tmp = data_x[ii][j]
            data_x[ii][j] = data_x[ii][len(data_x[ii]) - j - 1]
            data_x[ii][len(data_x[ii]) - j - 1] = tmp

    ax.plot_surface(data_x, data_y, data_z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'))
    plt.gca().view_init(elev=30, azim=30)
    plt.gca().set_zlim(-1.0, 1.0)
    plt.savefig(args.output + prefix + '.png')
    plt.close()
    plt.clf()


def plot_policy(env, args, x_range, y_range, policy, prefix):
    """
    Plots 2d graph depicting a policy. Right now I don't have any mechanism to break ties, I expect the user to have
    done that already. TODO: I don't support two actions at the same time, so I'll always depict only one action. I
    should support this in the future.

    :param env: object containing the environment's features, I access it to plot what I want
    :param args: list of all arguments passed at the execution so we can access the paths we are interested at
    :param x_range: one of the dimensions of the grid world
    :param y_range: the other dimension of the grid world
    :param policy: actual policy to be plotted. It should be encoded as an integer per state.
    :param prefix: string that will be used as the filename of the generated graph
    """
    plt.clf()
    plt.close()

    from pylab import rcParams
    rcParams['figure.figsize'] = y_range, x_range

    plt.xlim([0, y_range])
    plt.ylim([0, x_range])

    for i in range(y_range):
        plt.axvline(i, color='k', linestyle=':')
    plt.axvline(y_range, color='k', linestyle=':')

    for j in range(x_range):
        plt.axhline(j, color='k', linestyle=':')
    plt.axhline(x_range, color='k', linestyle=':')

    for idx in range(len(policy)):
        i, j = env.get_state_xy(idx)

        dx = 0
        dy = 0
        if policy[idx] == 0:  # up
            dy = 0.001
        elif policy[idx] == 1:  # right
            dx = 0.001
        elif policy[idx] == 2:  # down
            dy = -0.001
        elif policy[idx] == 3:  # left
            dx = -0.001

        if (env._matrix_mdp[i][j] != -1 and policy[idx] == 4) or (env._goal_x == i and env._goal_y == j):  # termination
            termination = plt.Rectangle(
                (j, x_range - i - 1), 1, 1, color='r')
            plt.gca().add_artist(termination)

        elif env._matrix_mdp[i][j] != -1:
            plt.arrow(j + 0.5 - 250 * dx, x_range - i + 0.5 - 1 - 250 * dy, dx, dy,
                      head_width=0.2, head_length=0.5, fc='k', ec='k')
        else:
            plt.gca().add_patch(
                patches.Rectangle(
                    (j, x_range - i - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor="gray"
                )
            )

    plt.savefig(args.output + prefix + '.png')
    plt.clf()


def plot_learning_curve(data, prefix):
    """
    Plot a regular learning curve with confidence intervals.

    :param data: Result of multiple runs of the algorithm.
    :param prefix: string that will be used as the filename of the generated graph
    """
    mean, lower_bound, upper_bound = utils.mean_confidence_interval(data)
    x_lim = len(data[0])
    plt.plot(mean, color='red')
    plt.fill_between(range(x_lim), lower_bound, upper_bound, color='red', alpha=0.4)
    plt.xlabel("Episode")
    plt.ylabel("Avg. number of steps")
    plt.savefig(prefix + 'learning_curve.png')
    plt.clf()
