from abstractagent import AbstractAgent
from blackjack import BlackjackObservation, BlackjackEnv, BlackjackAction
from collections import defaultdict
from carddeck import *
def get_started():
    return defaultdict(lambda :0)
    # tmp = {}
    # for i in range

class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function
    and the get_hypothesis() method that returns the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal difference method.
    """
    U = get_started()
    Ns = get_started()

    def train(self):
        for i in range(self.number_of_episodes):
            #print(i)
            observation = self.env.reset()
            terminal = False
            reward = 0
            current_state = (observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))
            #self.Ns[current_state]+=1
            while not terminal:
                previous_state = current_state
                # render method will print you the situation in the terminal
                # self.env.render()
                action = self.receive_observation_and_get_action(observation, terminal)
                observation, reward, terminal, _ = self.env.step(action)
                current_state = (observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))
                self.Ns[current_state]+=1
                learning_rate = 30/(29+self.Ns[previous_state])
                self.U[previous_state]+= learning_rate*(reward+0.9*self.U[current_state]-self.U[previous_state])

            # self.env.render()


    def receive_observation_and_get_action(self, observation: BlackjackObservation, terminal: bool) -> int:
        return BlackjackAction.HIT.value if observation.player_hand.value() < 17 else BlackjackAction.STAND.value

    def get_hypothesis(self, observation: BlackjackObservation, terminal: bool) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned U value for
        particular observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param terminal: Whether the hands were seen after the end of the game, i.e. whether the state is terminal.
        :return: The learned U-value for the given observation.
        """
        return self.U[(observation.player_hand.value(), observation.dealer_hand.value(), self.find_ace(observation))]
