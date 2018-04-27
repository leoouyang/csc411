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

    def play_second_against_random(self, action, print_grid=False):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if print_grid:
            env.render()
        if not done and self.turn == 1:
            state, s1, done = self.random_step()
            if print_grid:
                env.render()
            if done:
                if s1 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s1 == self.STATUS_TIE:
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
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim = 1)
        )
        self.cuda()

    def forward(self, x):
        return self.net(x)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state).type(torch.cuda.FloatTensor))
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
            Environment.STATUS_INVALID_MOVE: -10,
            Environment.STATUS_WIN         : 5,
            Environment.STATUS_TIE         : 1,
            Environment.STATUS_LOSE        : -5
    }[status]

def play_second_against_random(state, env):
    """a random agent play a move, and then the policy decide the next move."""
    done = False
    status = env.STATUS_VALID_MOVE
    #this value won't be actually used. Just to occupy the place
    if env.turn == 1:
        state, s1, done = env.random_step()
        if done:
            if s1 == env.STATUS_WIN:
                status = env.STATUS_LOSE
            elif s1 == env.STATUS_TIE:
                status = env.STATUS_TIE
            else:
                raise ValueError("???")
    return state, status, done

def train(policy, env, gamma=1.0, log_interval=1000, maxIter=60000, plot_curve=False):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.0001)
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
        if i_episode % 2 == 0:
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
        else:
            state, status, done = env.random_step()
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_second_against_random(action)
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

def first_player(state, policy, env):
    if env.turn == 1:
        action, logprob = select_action(policy, state)
        state, status, done = env.step(action)

def train_against_self(policy, env, gamma=1.0, log_interval=1000, maxIter=60000, plot_curve=False):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.0001)
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
        if i_episode % 2 == 0:
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.step(action)
                if not done and env.turn == 2:
                    status2 = env.STATUS_INVALID_MOVE
                    while status2 == env.STATUS_INVALID_MOVE:
                        action2, logprob2 = select_action(policy, state)
                        state, status2, done = env.step(action2)
                    if done:
                        if status2 == env.STATUS_WIN:
                            status = env.STATUS_LOSE
                        elif status2 == env.STATUS_TIE:
                            status = env.STATUS_TIE
                        else:
                            raise ValueError("???")
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
        else:
            status1 = env.STATUS_INVALID_MOVE
            while status1 == env.STATUS_INVALID_MOVE:
                action1, logprob1 = select_action(policy, state)
                state, status1, done = env.step(action1)
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.step(action)
                if not done and env.turn == 1:
                    status1 = env.STATUS_INVALID_MOVE
                    while status1 == env.STATUS_INVALID_MOVE:
                        action1, logprob1 = select_action(policy, state)
                        state, status1, done = env.step(action1)
                    if done:
                        if status1 == env.STATUS_WIN:
                            status = env.STATUS_LOSE
                        elif status1 == env.STATUS_TIE:
                            status = env.STATUS_TIE
                        else:
                            raise ValueError("???")
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
                       "ttt-self/policy-%d.pkl" % i_episode)

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


def load_weights(policy, episode, againstself = False):
    """Load saved weights"""
    if againstself:
        weights = torch.load("ttt-self/policy-%d.pkl" % episode)
    else:
        weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


def perform(episode, policy, env, times, first, printf = False, againstself = False):
    win, lose, tie = 0,0,0
    invalid = 0

    load_weights(policy, episode, againstself=againstself)
    for game in range(times):
        state = env.reset()
        done = False
        if first:
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
                if status == env.STATUS_INVALID_MOVE:
                    invalid += 1
        else:
            while not done:
                state, status, done = play_second_against_random(state, env)
                if not done:
                    action, logprob = select_action(policy, state)
                    state, status, done = env.step(action)
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

def display_against_random(episode, policy, env, times, againstself = False):
    load_weights(policy, episode, againstself=againstself)
    for game in range(times):
        print('\n===== Game' + str(game + 1) + ' =====')
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, print_grid=True)
            if done:
                print('Learned policy ' + status + 's against random! (learned policy moves first)')


def display_second_against_random(episode, policy, env, times, againstself = False):
    load_weights(policy, episode, againstself=againstself)
    for game in range(times):
        print('\n===== Game' + str(game + 1) + ' =====')
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_second_against_random(action, print_grid=True)
            if done:
                print('Learned policy ' + status + 's against random! (random moves first)')


def display_self_play(episode, policy, env, times, againstself = False):
    load_weights(policy, episode, againstself=againstself)
    for game in range(times):
        print('\n===== Game' + str(game + 1) + ' =====')
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.step(action)
            env.render()
            if done:
                if status == env.STATUS_WIN:
                    print('First player wins!')
                elif status == env.STATUS_TIE:
                    print('Tie.')
            if not done and env.turn == 2:
                status2 = env.STATUS_INVALID_MOVE
                while status2 == env.STATUS_INVALID_MOVE:
                    action2, logprob2 = select_action(policy, state)
                    state, status2, done = env.step(action2)
                    env.render()
                if done:
                    if status2 == env.STATUS_WIN:
                        status = env.STATUS_LOSE
                        print('Second player wins!')
                    elif status2 == env.STATUS_TIE:
                        status = env.STATUS_TIE
                        print('Tie.')
                    else:
                        raise ValueError("???")

if __name__ == '__main__':
    torch.manual_seed(1)
    random.seed(1)

    policy = Policy()
    env = Environment()

    train_max_iter = 100000
    test_games = 400
    discount_gamma = 0.8

    print("===============================Part 1==============================")
    torch.manual_seed(1)
    random.seed(1)
    policy = Policy(hidden_size=256)
    train(policy, env, gamma=discount_gamma, maxIter = train_max_iter, plot_curve=True)
    torch.manual_seed(1)
    random.seed(1)
    wins1 = []
    # winsties1 = []
    wins2 = []
    # winsties2 = []
    episodes = range(1000, train_max_iter + 1, 1000)
    for episode in episodes:
        win1, lose, tie1, invalid1 = perform(episode, policy, env, test_games, first=True)
        win2, lose, tie2, invalid2 = perform(episode, policy, env, test_games, first=False)
        wins1.append(win1/test_games)
        wins2.append(win2/test_games)
        # winsties1.append((win1+tie1)/test_games)
        # winsties2.append((win2+tie2)/test_games)
        # print("=====================episode %i============================" % (episode))
        # print(invalid1)
        # print(invalid2)
    plt.plot(episodes, wins1)
    plt.plot(episodes, wins2)
    # plt.plot(episodes, winsties1)
    # plt.plot(episodes, winsties2)
    # plt.legend(['Move first', 'Move second','wt1','wt2'],
    #            loc='upper left')
    plt.legend(['Move first', 'Move second'],
               loc='upper left')
    plt.axis([0, train_max_iter, 0, 1.1])
    plt.ylabel("Win rate")
    plt.xlabel("Episodes")
    plt.show()

    average_wins = []
    for i in range(len(wins1)):
        average_wins.append((wins1[i]+wins2[i])/2)
    best_episode = episodes[average_wins.index(max(average_wins))]
    print("Perform best at episode", best_episode)

    policy = Policy(hidden_size=256)
    display_against_random(best_episode, policy, env, 2)
    display_second_against_random(best_episode, policy, env, 3)
    print("===============================Part 2==============================")
    torch.manual_seed(1)
    random.seed(1)
    policy = Policy(hidden_size=256)
    load_weights(policy,best_episode)
    train_against_self(policy, env, gamma=0.9, maxIter = train_max_iter, plot_curve=True)
    torch.manual_seed(1)
    random.seed(1)
    wins1 = []
    winsties1 = []
    wins2 = []
    winsties2 = []
    episodes = range(1000, train_max_iter + 1, 1000)
    for episode in episodes:
        win1, lose, tie1, invalid1 = perform(episode, policy, env, test_games, first=True, againstself=True)
        win2, lose, tie2, invalid2 = perform(episode, policy, env, test_games, first=False, againstself=True)
        wins1.append(win1/test_games)
        wins2.append(win2/test_games)
        winsties1.append((win1+tie1)/test_games)
        winsties2.append((win2+tie2)/test_games)
        # print("=====================episode %i============================" % (episode))
        # print(invalid1)
        # print(invalid2)
    plt.plot(episodes, wins1)
    plt.plot(episodes, winsties1)
    plt.legend(['Win rate','Win & tie rate'],
               loc='upper left')
    plt.axis([0, train_max_iter, 0, 1.1])
    plt.ylabel("Percentage")
    plt.xlabel("Episodes")
    plt.show()

    plt.plot(episodes, wins2)
    plt.plot(episodes, winsties2)
    plt.legend(['Win rate','Win & tie rate'],
               loc='upper left')
    plt.axis([0, train_max_iter, 0, 1.1])
    plt.ylabel("Percentage")
    plt.xlabel("Episodes")
    plt.show()

    display_against_random(60000, policy, env, 2, againstself = True)
    display_second_against_random(60000, policy, env, 3, againstself = True)
    display_self_play(60000, policy, env, 5, againstself = True)
