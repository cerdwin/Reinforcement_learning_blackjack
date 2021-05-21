from gym.wrappers import Monitor

from abstractagent import AbstractAgent
from dealeragent import DealerAgent
from evaluate import *
import gym
from gym import wrappers
from gym.envs.registration import register
from randomagent import RandomAgent
from sarsaagent import SarsaAgent
from tdagent import TDAgent
from advancedagent import AdvancedAgent


def get_env() -> Monitor:
    """
    Creates the environment. Check the OpenAI Gym documentation.

    :rtype: Environment of the blackjack game that follows the OpenAI Gym API.
    """
    environment = gym.make('smu-blackjack-v0')
    return wrappers.Monitor(environment, 'smuproject4', force=True, video_callable=False)


if __name__ == "__main__":
    # Registers the environment so that it can be used
    register(
        id='smu-blackjack-v0',
        entry_point='blackjack:BlackjackEnv'
    )
    # ######################################################
    # IMPORTANT: do not modify the code above this line! ###
    # ######################################################

    # here you can play with the code yourself
    # for example you may want to split the code to two phases - training and testing
    # or you may want to compare two agents
    # feel free to modify the number of games played (highly recommended!)
    # ... or whatever

    env = get_env()
    number_of_episodes = 1000000  # TODO do not forget to change the number of episodes

    #agent: AbstractAgent = RandomAgent(env, number_of_episodes)
    #agent: AbstractAgent = DealerAgent(env, number_of_episodes)
    #agent: AbstractAgent = TDAgent(env, number_of_episodes)
    agent: AbstractAgent = SarsaAgent(env, number_of_episodes)
    #agent: AbstractAgent = AdvancedAgent(env, number_of_episodes)
    agent.train()

    # in evaluate.py are some ideas that you might want to use to evaluate the agent
    # feel free to modify the code as you want to
    # have club nine, diamond jack and spades two in your hand, and dealer has club four?
    # BlackjackObservation(copy.deepcopy(self.player_hand), copy.deepcopy(self.dealer_hand))
    evaluate(env.get_episode_rewards())
    #What is the utility for drawing a card when you have club nine, diamond jack and spades two in your hand, and dealer has club four?
    print('first Q is:', agent.Q[(21, 4, False)])
    # What is the utility of the situation when you have diamond ace and spades five and dealer spades ace?
    print('second Q is:', agent.Q[(16, 11, False)])

    print('Q is:', agent.Q)

