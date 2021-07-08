[//]: # (Image References)

[image1]: ./training_scores.png?raw=true "Training Score"

### 0. Introduction
This project is to create an agent within Unity's banana collector environment to achieve optimal score based on following rules: collecting a yellow banana receives a reward of +1; collecting a blue banana recevies a reward of -1.
In order to solve the environment, our agent must achieve an average score of +13 over 100 consecutive episodes.
We applied reinforcement learning algorithm to train the agent to complete this task. The general idea is to obtain a policy that helps agent to select the optimal action based on the given state from the environment. The policy is learned by the agent in the process of trial-and-error. Starting from selecting random actions in the action universe, the agents collects the reward obtained from each action under each state, then the agent can select the optimal action (i.e. action with largest reward) under the given state. Thus, the process of finding the optimal policy can be transformed to estimating the optimal action-value function Q*(s,a) that provides the expected reward for each state s and action a. Here we estimated this action-value function with a Q-Network, also applied some additional methods to improve the stability/accuracy of the algorithm.

### 1. Learning Algorithm
We've discussed that the goal is to achieve an estimation of optimal action-value function. There are several ways to achieve this. It can be solved iteratively during the learning process using Bellman Equation, however this turns out to be impractical, because this action-value function is estimated separately for each sequence without any generalization. It's more common to use a function approximator to estimate the action-value function:
<a href="https://www.codecogs.com/eqnedit.php?latex=Q^*(s,a;\theta)&space;\approx&space;Q^*(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^*(s,a;\theta)&space;\approx&space;Q^*(s,a)" title="Q^*(s,a;\theta) \approx Q^*(s,a)" /></a>
Specifically, we refer to a neural network function approximator with weights theta as a Q-Network. A Q-Network can be trained by adjusting the parameter theta_i at iteration i to reduce the mean-square error in the Bellman Equation. We applied the DQN algorithm to train the parameters of this Q-Network and obtain the estimation of optimal action-value function.

#### Network Structure
The basic structure of Q-Network contains 3 layers, which are all fully-connected layers with 512, 1024, and 4 nodes separately. (Please refer to model.py for details) Number of layers, and the number of nodes in each layer are hyperparameters in this model. We can also apply convolutional neural-network layers, but in this case, the input we get from environment is one-dimensional, which is not the best use case for convolutional layers. We can apply CNN algorithms when we train the network with pixels (when we take the 2-d pixel matrix as input to the network).
Also, when training the network, we select action from state and the existing action-value function Q based on `𝛜`-greedy algorithm. Specifically, we choose a random action within the action universe with probablity `𝛜`, and we choose the action with maximum Q value in the given state with probablity 1-`𝛜`. This is to address the exploration-exploitation dilemma. Moreover, the value of `𝛜` being used is decaying in the process of training, so we increasingly favor exploitation as we are gaining more experience. The starting and ending values for `𝛜`, and the rate at which it decays are three hyperparameters in this model.

#### Modifications
Some methods are applied when training this Q-Network to increase the stability.
##### a. Experience replay
Instead of using the action/state/reward of each step in the training process to update the network, we choose to cache it in replay memory, then use a random data point from this memory to update the network. This improves the efficiency of data points, as it's potentially being used for muliple updates; Also, randomizing the samples breaks the correlation between samples and reduces the variance of the updates; Moreover, using experience replay makes the behavior distrubution being averaged over many of its previous states, smoothing out learning and avoiding oscillations or divergence in the parameters.

##### b. Separate network for generating target in the Q-learning update
We use a separate network to generate the target objective function value when updating the network parameters. (As objective function is related to the Q values of the next state) For every C updates, we clone the network Q to obtain a target network Q' and use Q' to generate the Q-learning targets in the update step. This makes the calculation more stable, as increasing Q(s_t, a) will also increase Q(s_t+1, a), so if we use the same network for target, we are actually letting the network to chase itself in the optimization process (like hanging a carrot in front of a donkey while we ride the donkey). Generating the target using an older set of parameters adds a delay between the time an update to Q is made and the time the update affects the targets, making divergence or oscillations much more unlikely.

#### Hyperparameters
Following hyperparameters are being applied in this model:
##### 1. Number of layers and number of nodes in each layer
This Q-Network uses 3 layers, and there are 512, 1024, and 4 nodes separately in each layer.
##### 2. Learning rate for the network update
The learning rate for updating the parameters is set to 5e-4.
##### 3. Discount rate for calculating expected future rewards
When calculating expected reward of actions, we need to apply a discount rate for the rewards in future time-stamps. Here the discount rate is set to 0.99.
##### 4. Replay buffer size and sample size
We apply a replay buffer with size of 1e5 when training the network, and everytime we select a mini-batch of 64 samples randomly from the memory buffer.
##### 5. Time steps between two updates of the network
For each 4 time steps, we do one update of the network in the training process.
##### 6. Interpolation parameter when soft updating the target network
When updating the target network with the local network parameters, we apply a interpolation parameter τ being used as: θ_target = τ*θ_local + (1 - τ)*θ_target. Here we apply τ = 1e-3.

### 2. Testing Result
The agent was able to successfully solve the environment within ~1100 episodes. Please see the graph below for the score achieved after each episode in the training process:

![Training Score][image1]

### 3. Future Improvements
##### a. Dueling Network
Dueling networks utilize two streams: one that estimates the state value function `V(s)`, and another that estimates the advantage for each action `A(s,a)`. These two values are then combined to obtain the desired Q-values. 
The reasoning behind this approach is that state values don't change much across actions, so it makes sense to estimate them directly. However, we still want to measure the impact that individual actions have in each state, hence the need for the advantage function.

##### b. Prioritized Replay
Currently we use the uniform sampling from the replay memory. A more sophisticated method would be we can emphasize transitions from which we can learn the most.

