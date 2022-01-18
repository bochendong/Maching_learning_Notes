# Fundamentals

## Core concepts
**Agents**: This could be a human being, a
robot, a game-player, a piece of risk management software, a portfolio
manager, etc. 

**Environment**: “The thing the agent interacts with and aims to solve”; the
problem space. This could be your entire life, a Go board, a chess board, an
Atari game, the sky (i.e. drones, fighter jets), a text conversation (chatbots) the stock market, etc.

**State**: A representation of the agent’s current situation. This can be a simple ID (State 1, State 2, …), coordinates, or a set of features. It can include compressed representations of the past (i.e., memory).

**Action**: Selected by the Agent, according to their Policy, that affect the Environment.

**Reward**: The special learning signal emitted from the Environment in response to Agent actions. Used by the Agent to learn about which actions and states are “good”. Rewards are usually specified by you.

**Timestep**: For this week, timesteps represent discrete action-reward-update cycles. Our learner takes an action, receives some reward, and updates its value estimates. Then we go to the next timestep.

**Run**: One run of N timesteps. We might conduct 1000 runs of 2000 timesteps

**Learning Rate**: Represented by the symbol $\alpha \in [0,1]$, learning rate is the weight on the most recent reward.

**Policy**: Represented by the symbol $\pi$, policies are how agents choose their actions. An example policy might be “always choose the action that I think gives me the best reward RIGHT NOW.” Examples of policies from this week are epsilon-greedy and
Upper-Confidence-Bound.