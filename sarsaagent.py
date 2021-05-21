from abstractagent import AbstractAgent
from blackjack import BlackjackEnv, BlackjackObservation
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from carddeck import *


def start_off():
    return defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: 0)


class SarsaAgent(AbstractAgent):
    """
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate) as long as you fulfil convergence criteria.
    """
    Q, Ns = start_off()
    N1 = 20
    N2 = 19


    def train(self):
        U1_STAND = [0 for i in range(self.number_of_episodes)]  # stand
        U1_HIT = [0 for i in range(self.number_of_episodes)]
        U2_STAND = [0 for i in range(self.number_of_episodes)]  #
        U2_HIT = [0 for i in range(self.number_of_episodes)]  #
        U = defaultdict(lambda: 0)
        discount = 0.9

        for i in range(self.number_of_episodes):
            observation = self.env.reset()
            terminal = False
            reward = 0
            current_state = (observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))
            self.Ns[current_state] += 1
            while not terminal:
                previous_state = current_state
                action = self.receive_observation_and_get_action(observation, terminal)
                observation, reward, terminal, _ = self.env.step(action)
                current_state = (observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))
                self.Ns[current_state] += 1
                previous_Q = self.Q[previous_state][action]
                current_Q = self.Q[current_state][self.receive_observation_and_get_action(observation, terminal)]
                self.Q[previous_state][action]+=self.N1/(self.N2+self.Ns[previous_state])*(reward+discount*current_Q-previous_Q)
                U1_STAND[i] = self.Q[(21, 4, False)][0]
                U1_HIT[i] = self.Q[(21, 4, False)][1]
                U2_STAND[i] = self.Q[(16, 11, False)][0]
                U2_HIT[i] = self.Q[(16, 11, False)][1]

        self.plot_STAND_series(U1_STAND,'U1_stand')
        self.plot_STAND_series(U2_STAND, 'U2_stand')
        self.plot_HIT_series(U1_HIT, 'U1_hit')
        self.plot_HIT_series(U2_HIT, 'U2_hit')


    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        '''
        implementation of e-greedy, as per Sutton's description, with a stepsize depending on
        how often was a state visited
        :param observation:
        :param terminal:
        :return:
        '''
        if random.random()>self.N1/(self.N2+self.Ns[(observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))]):
            arr = list(self.Q[(observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))] )
            # Maybe the following was too slow
            # return np.argmax(self.Q[(observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))])
            return arr.index(max(arr))
        return self.env.action_space.sample()



    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool, action: int) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned Q value for
        particular observation and action.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :param action: Action for Q-value.
        :return: The learned Q-value for the given observation and action.
        """
        return self.Q[(observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))][action]

    def plot_STAND_series(self, arr, figname):

        plt.plot(arr, c="b")
        plt.xlabel('Episodes')
        plt.ylabel('Utility of stand')
        plt.title('SARSA agent\'s utility')
        plt.grid(color='green', linestyle='--', linewidth=0.5)
        plt.savefig(figname)
        plt.show()

    def plot_HIT_series(self, arr, figname):

        plt.plot(arr, c="b")
        plt.xlabel('Episodes')
        plt.ylabel('Utility of hit')
        plt.title('SARSA agent\'s utility ')
        plt.grid(color='green', linestyle='--', linewidth=0.5)
        plt.savefig(figname)
        plt.show()