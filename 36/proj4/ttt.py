from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action, print_grid=False):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if print_grid:
            env.render()
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if print_grid:
                env.render()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ...

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    max_t = len(rewards)
    result = [0] * max_t
    for i in range(max_t):
        for j in range(i, max_t):
            result[i] += rewards[j] * (gamma**(j-i))
    return result


def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO
            Environment.STATUS_INVALID_MOVE: -15,
            Environment.STATUS_WIN         : 10,
            Environment.STATUS_TIE         : -1,
            Environment.STATUS_LOSE        : -10
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000, maxIter=60000, plot_curve=False):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    episodes = []
    avg_returns = []

    for i_episode in range(1, maxIter + 1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            # if status == env.STATUS_INVALID_MOVE:
            #     print("Invalid action:", action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            if plot_curve:
                avg_returns.append(running_reward / log_interval)
                episodes.append(i_episode)
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if plot_curve:
        plt.plot(episodes, avg_returns)
        plt.axis([0, maxIter, int(min(avg_returns)), int(max(avg_returns)) + 1])
        plt.ylabel("Average return")
        plt.xlabel("Episodes")
        plt.show()


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def perform(episode, policy, env, times, printf = False):
    win, lose, tie = 0,0,0
    invalid = 0

    load_weights(policy, episode)
    for game in range(times):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == env.STATUS_INVALID_MOVE:
                invalid += 1
        if status == env.STATUS_WIN:
            win += 1
        elif status == env.STATUS_LOSE:
            lose += 1
        else:
            tie += 1
    if printf:
        print("Use weights from episode", episode)
        print(times, "games played:")
        print("Win:", win, "Percentage:", win/times)
        print("Lose:", lose, "Percentage:", lose/times)
        print("Tie:", tie, "Percentage:", tie/times)
        print(invalid, "invalid actions are made")
    return win, lose, tie, invalid


def display(episode, policy, env, times):
    load_weights(policy, episode)
    for game in range(times):
        print('\n===== Game' + str(game + 1) + ' =====')
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, print_grid=True)
            if done:
                print('Learned policy ' + status + 's against random!')


if __name__ == '__main__':
    torch.manual_seed(1)
    random.seed(1)

    policy = Policy()
    env = Environment()

    train_max_iter = 60000
    test_games = 400
    discount_gamma = 0.8

    print("===============================Part 5a==============================")
    torch.manual_seed(1)
    random.seed(1)
    policy = Policy(hidden_size=64)
    train(policy, env, gamma=discount_gamma, maxIter = train_max_iter, plot_curve=True)

    print("===============================Part 5b==============================")
    best_hidden= None
    best_episode = None
    highest_win = 0
    hidden_range = [54, 64, 128, 256]
    for hidden_size in hidden_range:
        torch.manual_seed(1)
        random.seed(1)
        print("=======================================================")
        print(hidden_size)
        policy = Policy(hidden_size=hidden_size)
        train(policy, env, gamma=discount_gamma,maxIter=train_max_iter)
        wins = []
        winsAndTies = []
        episodes = range(1000,train_max_iter+1,1000)
        invalids = []
        for episode in episodes:
            win, lose, tie, invalid = perform(episode, policy, env, test_games)
            wins.append(win)
            winsAndTies.append(win+tie)
            invalids.append(invalid)
        best_i = wins.index(max(wins))
        if wins[best_i] > highest_win:
            highest_win = wins[best_i]
            best_episode = episodes[best_i]
            best_hidden = hidden_size
        print("Best performance at episode", episodes[best_i])
        print("Max win rate:", wins[best_i] / test_games * 100, "%")
        print("Max win and tie rate:", winsAndTies[best_i] / test_games * 100, "%")
        print("Invalid moves:",invalids[best_i])

    print("=======================================================")
    print("Training the best policy with", best_hidden, "hidden units")
    print("The policy perform best at", best_episode, "episode")
    torch.manual_seed(1)
    random.seed(1)
    policy = Policy(hidden_size=best_hidden)
    train(policy, env, gamma=discount_gamma, maxIter=train_max_iter)

    print("===============================Part 5c==============================")
    invalids = []
    episodes = range(1000, train_max_iter + 1, 1000)
    torch.manual_seed(1)
    random.seed(1)
    for episode in episodes:
        win, lose, tie, invalid = perform(episode, policy, env, test_games)
        invalids.append(invalid/test_games)
    plt.plot(episodes, invalids)
    plt.axis([0, train_max_iter, 0, int(max(invalids)) + 1])
    plt.ylabel("Average Invalid Moves/Episode")
    plt.xlabel("Episodes")
    plt.show()

    print("===============================Part 5d==============================")
    torch.manual_seed(1)
    random.seed(1)
    win, lose, tie, invalid = perform(best_episode, policy, env, 100, printf=True)
    display(best_episode, policy, env, 5)
    # change 49000 to best_episode afterwards TODO


    print("===============================Part 6==============================")
    episodes = range(1000, train_max_iter + 1, 1000)
    wins,loses,ties = [],[],[]
    torch.manual_seed(1)
    random.seed(1)
    for episode in episodes:
        win, lose, tie, invalid = perform(episode, policy, env, test_games)
        wins.append(win/test_games*100)
        loses.append(lose/test_games*100)
        ties.append(tie/test_games*100)
    plt.plot(episodes, wins)
    plt.plot(episodes, loses)
    plt.plot(episodes, ties)
    plt.legend(['win rate', 'loss rate','tie rate'],
               loc='upper left')
    plt.axis([0, train_max_iter, 0, 100])
    plt.ylabel("Percentage")
    plt.xlabel("Episodes")
    plt.show()

    print("===============================Part 7==============================")
    env.reset()

    best_episode = 49000
    best_hidden = 64

    load_weights(policy, best_episode)
    distr = first_move_distr(policy, env).tolist()[0]
    print("Distribution of the first step from policy with",best_hidden, "hidden units", "at episode", best_episode)
    for i in range(9):
        print("Probability playing position %i: %.2f %%" % (i, distr[i]))

    episodes = range(1000, train_max_iter + 1, 1000)
    distributions = [x[:] for x in [[]] * 9]
    for episode in episodes:
        load_weights(policy,episode)
        distr = first_move_distr(policy,env).tolist()[0]
        for i in range(9):
            distributions[i].append(distr[i]*100)
    legends = []
    for i in range(9):
        plt.plot(episodes, distributions[i])
        legends.append("Position %i" % i)

    plt.legend(legends,loc='upper left')
    plt.axis([0, train_max_iter, -5, 105])
    plt.ylabel("Percentage")
    plt.xlabel("Episodes")
    plt.show()
