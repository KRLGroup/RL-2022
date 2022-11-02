import numpy as np
import torch
import torch.nn as nn
from Q_network import Q_network
from utils import from_tuple_to_tensor

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=UserWarning)


class DDQN_agent:

    def __init__(self, env, rew_thre, buffer, learning_rate=0.001, initial_epsilon=0.5, batch_size= 64):

        self.env = env


        self.network = Q_network(env, learning_rate)
        #TODO
        #self.target_network = ???
        self.buffer = buffer
        self.epsilon = initial_epsilon
        self.batch_size = batch_size
        self.window = 50
        self.reward_threshold = rew_thre
        self.initialize()
        self.step_count = 0
        self.episode = 0


    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        #TODO
        #action = ??


        #simulate action
        s_1, r, done, _, _ = self.env.step(action)


        #put experience in the buffer
        #TODO
        # self.buffer ??

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:

            self.s_0, _ = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=10000,
              network_update_frequency=10,
              network_sync_frequency=100):
        self.gamma = gamma

        self.loss_function = nn.MSELoss()
        self.s_0, _ = self.env.reset()

        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')
        ep = 0
        training = True
        self.populate = False
        while training:
            self.s_0, _ = self.env.reset()

            self.rewards = 0
            done = False
            while not done:
                if ((ep % 5) == 0):
                    self.env.render()

                p = np.random.random()
                if p < self.epsilon:
                    done = self.take_step(mode='explore')
                    # print("explore")
                else:
                    done = self.take_step(mode='exploit')
                    # print("train")
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    # TODO: synchronize Qnet and target_net
                    #self.target_network ???
                    self.sync_eps.append(ep)

                if done:
                    if self.epsilon >= 0.05:
                        self.epsilon = self.epsilon * 0.7
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    mean_loss = np.mean(self.training_loss[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}   mean loss = {:.2f} \t\t".format(
                            ep, mean_rewards, self.rewards, mean_loss), end="")

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        #break
        # save models
        self.save_models()
        # plot
        self.plot_training_rewards()

    def save_models(self):
        torch.save(self.network, "Q_net")

    def load_models(self):
        self.network = torch.load("Q_net")
        self.network.eval()

    def plot_training_rewards(self):
        plt.plot(self.mean_training_rewards)
        plt.title('Mean training rewards')
        plt.ylabel('Reward')
        plt.xlabel('Episods')
        plt.show()
        plt.savefig('mean_training_rewards.png')
        plt.clf()

    def calculate_loss(self, batch):
        #extract info from batch
        states, actions, rewards, dones, next_states = list(batch)

        #transform in torch tensors
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        actions = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(device)
        states = from_tuple_to_tensor(states)
        next_states = from_tuple_to_tensor(next_states)

        ###############
        # DDQN Update #
        ###############
        #TODO
        # Q(s,a) = ??
        #
        #

        # TODO
        # target Q(s,a) = ??
        #
        #
        #


        #TODO
        #loss = self.loss_function( Q(s,a) , target_Q(s,a))
        loss = 0

        return loss


    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        loss.backward()
        self.network.optimizer.step()

        if device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0

    def evaluate(self, eval_env):
        done = False
        s, _ = eval_env.reset()
        rew = 0
        while not done:
            action = self.network.greedy_action(torch.FloatTensor(s))
            s, r, done, _, _ =eval_env.step(action)
            rew += r

        print("Evaluation cumulative reward: ", rew)
