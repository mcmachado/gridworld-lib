# Author Marlos C. Machado

import sys
import numpy as np


class GridWorld:
    _str_mdp = ''
    _num_rows = -1
    _num_cols = -1
    _num_states = -1
    _matrix_mdp = None
    _adj_matrix = None
    _reward_function = None
    _use_negative_rewards = False

    _curr_x = 0
    _curr_y = 0
    _start_x = 0
    _start_y = 0
    _goal_x = 0
    _goal_y = 0

    def __init__(self, path=None, str_in=None, use_negative_rewards=False):
        """
        Return a GridWorld object that instantiates the MDP defined in a file (specified in path). In case it is None,
        then the MDP definition is read from str_in, which is a string with the content that path would hold. The input
        should have a very specific format. The first line should contain two numbers separated by a comma. These
        numbers define the dimensions of the MDP. The rest of the lines are composed of X's denoting walls and of .'s
        denoting empty spaces in the MDP. S denotes the starting state.

        :param path:
        :param str_in:
        :param use_negative_rewards:
        """
        if path is not None:
            self._read_file(path)
        elif str_in is not None:
            self._str_mdp = str_in
        else:
            print("You are supposed to provide an MDP specification as input!")
            sys.exit()

        self._parse_string()
        self._curr_x = self._start_x
        self._curr_y = self._start_y
        self._num_states = self._num_rows * self._num_cols
        self._use_negative_rewards = use_negative_rewards

    def _read_file(self, path):
        """
        We just read the file and put its contents in strMDP.

        :param path: path to the file containing the MDP description
        """
        file_name = open(path, 'r')
        for line in file_name:
            self._str_mdp += line

    def _parse_string(self):
        """
        I now parse the received string. I'll store everything in a matrix (matrixMDP) such that -1 means wall and 0
        means available square. The letter 'S' is converted to the initial (x,y) position.

        :return:
        """
        data = self._str_mdp.split('\n')
        self._num_rows = int(data[0].split(',')[0])
        self._num_cols = int(data[0].split(',')[1])
        self._matrix_mdp = np.zeros((self._num_rows, self._num_cols))

        for i in range(len(data) - 1):
            for j in range(len(data[i + 1])):
                if data[i + 1][j] == 'X':
                    self._matrix_mdp[i][j] = -1
                elif data[i + 1][j] == '.':
                    self._matrix_mdp[i][j] = 0
                elif data[i + 1][j] == 'S':
                    self._matrix_mdp[i][j] = 0
                    self._start_x = i
                    self._start_y = j
                elif data[i + 1][j] == 'G':
                    self._matrix_mdp[i][j] = 0
                    self._goal_x = i
                    self._goal_y = j

    def _fill_adjacency_matrix(self):
        """
        Given a description of a grid world, we can generate the adjacency matrix of the underlying graph representing
        it. This function does exactly that. For each state, it looks at the four states around it (up, down, left,
        right) to see if these are valid states. If they are, an edge representing that transition is added to the
        adjacency matrix. Notice this implementation assumes every transition can be reversed.
        """
        self._adj_matrix = np.zeros((self._num_states, self._num_states), dtype=np.int)
        self.idxMatrix = np.zeros((self._num_rows, self._num_cols), dtype=np.int)

        # I'll try for all states not in the borders (they have to be walls) all 4 possible directions.
        # If the next state is also available we add such entry to the adjacency matrix, otherwise we don't.
        # This is not necessarily efficient, but for small MDPs it should be fast enough.
        for i in range(len(self.idxMatrix)):
            for j in range(len(self.idxMatrix[i])):
                self.idxMatrix[i][j] = i * self._num_cols + j

        for i in range(len(self._matrix_mdp)):
            for j in range(len(self._matrix_mdp[i])):
                if i != 0 and i != (self._num_rows - 1) and j != 0 and j != (self._num_cols - 1):  # Ignore the borders
                    if self._matrix_mdp[i + 1][j] != -1:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i + 1][j]] = 1
                    else:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j]] += 1  # Considering self-loops
                    if self._matrix_mdp[i - 1][j] != -1:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i - 1][j]] = 1
                    else:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j]] += 1  # Considering self-loops
                    if self._matrix_mdp[i][j + 1] != -1:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j + 1]] = 1
                    else:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j]] += 1  # Considering self-loops
                    if self._matrix_mdp[i][j - 1] != -1:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j - 1]] = 1
                    else:
                        self._adj_matrix[self.idxMatrix[i][j]][self.idxMatrix[i][j]] += 1  # Considering self-loops

    def _get_state_idx(self, x, y):
        """
        Given a state coordinate (x,y) this method returns the index that uniquely identifies this state.

        :param x: value of the coordinate x
        :param y: value of the coordinate y
        :return : unique index identifying a position in the grid
        """
        idx = y + x * self._num_cols
        return idx

    def _get_next_state(self, action):
        """
        This function returns what was going to be the next state x,y if the action given as parameter was to be taken.
        It does not update the next state, it is a one-step forward model.

        :param action: action to be taken
        :return: x, y values corresponding to what would be the agent's next state if the action was to be taken
        """
        next_x = self._curr_x
        next_y = self._curr_y

        if self._matrix_mdp[self._curr_x][self._curr_y] != -1:
            if action == 'up' and self._curr_x > 0:
                next_x = self._curr_x - 1
                next_y = self._curr_y
            elif action == 'right' and self._curr_y < self._num_cols - 1:
                next_x = self._curr_x
                next_y = self._curr_y + 1
            elif action == 'down' and self._curr_x < self._num_rows - 1:
                next_x = self._curr_x + 1
                next_y = self._curr_y
            elif action == 'left' and self._curr_y > 0:
                next_x = self._curr_x
                next_y = self._curr_y - 1

        if next_x < 0 or next_y < 0:
            print("You were supposed to have hit a wall before!\nThere is something wrong with your MDP definition.")
            sys.exit()

        if next_x == len(self._matrix_mdp) or next_y == len(self._matrix_mdp[0]):
            print("You were supposed to have hit a wall before!\nThere is something wrong with your MDP definition.")
            sys.exit()

        if self._matrix_mdp[next_x][next_y] != -1:
            return next_x, next_y
        else:
            return self._curr_x, self._curr_y

    def _get_next_reward(self, next_x, next_y):
        """
        Returns the reward the agent will observe if in state (currX, currY) and it takes action 'action' leading to
        the state (nextX, nextY). As _get_next_state, this is a one-step model because it responds with the reward the
        agent would see if going from any state to any other adjacent state. It is not only about the agent's current
        state. NOTICE THIS FUNCTION DOES NOT CHECK WHETHER SUCH TRANSITION IS VALID.

        :param next_x: state the agent would land in (x-value)
        :param next_y: state the agent would land in (y-value)
        :return: reward signal the agent would observe
        """
        # If we want to observe negative rewards at every time step until reaching the goal state, we just check
        # If the landing state is invalid, or if it is the absorbing state. If it is return 0, otherwise return -1
        if self._use_negative_rewards:
            if self._matrix_mdp[next_x][next_y] == -1 or self._get_state_idx(next_x, next_y) == self._num_states:
                return 0
            else:
                return -1
        else:  # Otherwise return 1 if the agent reaches the goal state and 0 otherwise
            if next_x == self._goal_x and next_y == self._goal_y:
                return 1
            else:
                return 0

    @staticmethod
    def get_action_set():
        """
        I'm only supporting the four directional actions for now.

        :return: action set
        """
        return ['up', 'right', 'down', 'left']

    def get_num_states(self):
        """
        Returns the total number of states (including walls) in the MDP.

        :return: num states
        """
        return self._num_states

    def get_grid_dimensions(self):
        """
        Returns grid world width and height.

        :return: dimensions of the grid the agent lives in (rows and columns)
        """
        return self._num_rows, self._num_cols

    def get_state_xy(self, idx):
        """
        Given the index that uniquely identifies each state this method returns its equivalent coordinate (x,y).

        :param idx: index uniquely identifying a state
        :return: values x, y describing the state's location in the grid
        """
        y = int(idx % self._num_cols)
        x = int((idx - y) / self._num_cols)
        return x, y

    def get_current_state(self):
        """
        Returns the unique identifier for the current state the agent is.

        :return: Index representing the agent's current state.
        """
        curr_state_idx = self._get_state_idx(self._curr_x, self._curr_y)
        return curr_state_idx

    def act(self, action):
        """
        There are four possible actions: up, down, left and right. If the agent tries to go to a -1 state (wall) it
        will stay on the same coordinate. I decided to implement everything being deterministic for now.

        :param action: action to be taken by the agent
        :return: reward generated by the environment
        """
        # I get what will be the next state and before really making it my current state I verify everything is sound.
        if self.is_terminal():
            return 0
        else:
            next_x, next_y = self._get_next_state(action)  # We check if the transition is sound in this function
            reward = self._get_next_reward(next_x, next_y)
            self._curr_x = next_x
            self._curr_y = next_y
            return reward

    def reset(self):
        """
        Resets the agent to its initial position.
        """
        self._curr_x = self._start_x
        self._curr_y = self._start_y

    def is_terminal(self):
        """
        Checks whether the agent is in a terminal state (or goal).

        :return: true if the agent's current state is a goal state, otherwise return false
        """
        if self._curr_x == self._goal_x and self._curr_y == self._goal_y:
            return True
        else:
            return False

    def get_adjacency_matrix(self):
        """
        If I never did it before, I will fill the adjacency matrix. Otherwise I'll just return the one I already have.

        :return: adjacency matrix representing the loaded MDP
        """
        if np.all(self._adj_matrix is None):
            self._fill_adjacency_matrix()
        return self._adj_matrix

    def get_next_state_and_reward(self, curr_state, action):
        """
        One step forward model: return the next state and reward the agent would observe if they were in a particular
        state and took a specific action.

        :param curr_state: Start state the agent would be at
        :param action: Action the agent would take
        :return: index of the state the agent would end up at, and the reward it would end up observing
        """
        # In case it is the absorbing state encoding end of an episode
        if curr_state == self._num_states:
            return curr_state, 0

        # Now I can reset the agent to the state I was told to
        temp_x = self._curr_x
        temp_y = self._curr_y
        self._curr_x, self._curr_y = self.get_state_xy(curr_state)

        reward = -1  # dummy initialization
        next_state_idx = -1  # dummy initialization
        # Now I can ask what will happen next in this new state
        if self.is_terminal():
            next_state_idx = self._num_states
            reward = 0
        else:
            next_x, next_y = self._get_next_state(action)
            if next_x != -1 and next_y != -1:  # If it is not the absorbing state:
                reward = self._get_next_reward(next_x, next_y)
                next_state_idx = self._get_state_idx(next_x, next_y)

        # We need to restore the previous configuration:
        self._curr_x = temp_x
        self._curr_y = temp_y

        return next_state_idx, reward
