   # Table of Contents

1. Introduction
   1. Motivation
      - General discussion about autonomous driving
      - General discussion about reinforcement learning and recent results with Deep Reinforcement Learning
      - The increasing interests in real-world problems and not simulations
      - Focus on the object of the thesis, motivation, description of the procedure followed and results obtained
   2. Structure of the Thesis
      - Brief description of every chapter
2. Background and Related Work
   - Introduction to the chapter
   1. Elements of Reinforcement Learning
      - Introduction
      - Central concepts regarding fundamentals (e.g. Agent, Environment, Reward, Return)
      - Markov Decision Process (MDP)
      - Model-based prevision and control: just a brief introduction to Dynamic Programming and Policy/Value Iteration
      - Model-free prevision and control: a brief introduction to Monte Carlo and TD Learning approaches (SARSA and Q-Learning)
   2. Deep Reinforcement Learning
      - Introduction and motivation behind function approximation
      - Value-based methods and Policy Gradient methods
      - Focus and detailed explanation of DDPG and SAC
   3. Related Work
      - Explanation of the state-of-the-art focusing more on Reda's paper, its approach and the related bibliography.
      - Increasing interest in Reinforcement Learning applied to real-world situations, in contrast with simulated environment experiments.
3. Tools and Frameworks
   - Introduction to the chapter
   1. Open AI Gym
      - General description of the framework
      - Discussion about the motivation and the importance of this framework, such as the necessity of a Reinforcement Learning Framework to test different algorithms with different environments providing the same interface.
      - One contribution of the thesis: the creation and design of an OpenAI Gym environment for Anki Cozmo Environment with connection to chapter 4.
   2. Anki Cozmo
      - General description of Cozmo and Anki
      - General information about the mechanics and features of Cozmo ([LINK](https://www.cs.cmu.edu/afs/cs/academic/class/15494-s17/schedule.html))
      - Discussion about the selection of Cozmo instead of other solutions
        - Amazon Deep Racer: not available at the start of the thesis. It provides a simulator to train the agent. Using AWS for computation which can be a benefit, but also a drawback because it is a lock-in solution.
        - Building a Car: one of the best path to follow because of the personalization available. Main drawbacks are the length of the car construction process but also the time to spend in the creation of interfaces between the car and Python.
        - In the end, Cozmo is the best trade-off between functionalities and fast-developing. It provides plain and straightforward control of the car and a rich Python SDK to use with OpenAI Gym.
      - Discussion about the on-board or off-board computation
   3. PyTorch
      - General Description of Pytorch framework.
      - Why Pytorch? Numpy-like framework and TensorFlow 2.0 not yet released
4. Design of the control system
   - Introduction to the chapter
   1. General description of the control systems with a diagram representing all the technologies involved.
   2. Setup of the algorithms (DDPG, SAC)
      - We decided to rewrite both algorithms without using libraries directly. The first motivation was didactical, implementing from scratch is helpful to understand the practical implementation of the algorithm better, but also to make it possible to implement the singularity of the real Cozmo environment.
      - The interaction with OpenAI Gym and PyTorch
      - Discussion about Hyper-Parameters and the problems faced in the real world situation in the selection of these parameters.
   3. Setup and implementation of CozmoEnv
      - Technologies used to implement the interaction between the Cozmo SDK and OpenAI Gym.
      - Differences from the simulated environment caused by the need for direct human interaction.
      - Implementation of human interaction in the system.
   4. Setup of the real Environment
      - The Track design
      - Analysis of the problems:
        - Reflection
        - Background and Horizon
      - (Single Line Track)
5. Experimental results
   1. Introduction
      - Arguments of the chapter
      - Overview of final results
   2. Experimental Methodology
      - Preliminaries: experiments with Pendulum-v0
      - Real Life experiments with Cozmo
      - Hyper-Parameters discussion and motivation
      - Algorithms applied and modifications with pseudo-code
   3. Experiments with Pendulum-v0
      - Comparative analysis between results obtained with DDPG and SAC
      - Results
   4. Experiments with Cozmo
      - Comparative analysis between results obtained with DDPG and SAC
      - Results
6. Conclusions
    - Comments on the results obtained
    - Strong points and weaknesses
    - Autocriticism about weaknesses that affect the final result
   1. Future Work
      - Data efficiency and Model-based approaches through a better study of bibliography
      - Usage of Variational Auto Encoder (VAE): this is a possible additional step towards the creation of the model (VAE used for data generation as a generative model). It must not overwrite the first future work.
      - New Version of Cozmo (Vector) with a better camera and LIDAR (Data fusion);
