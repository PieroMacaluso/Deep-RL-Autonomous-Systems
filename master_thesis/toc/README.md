# Table of Contents

1. Introduction
   1. Motivation
   2. Structure of the Thesis
      - Brief description of every chapter
2. Background and Related Work
   - Introduction to the chapter
   1. Elements of Reinforcement Learning
      - Introduction
      - Main concept regarding fundamentals (Agent, Environment, Reward,...)
      - Model-based prevision and control: just a brief introduction to Dynamic Programming and Policy/Value Iteration
      - Model-free prevision and control: brief introduction to Monte Carlo and TD Learning approaches (SARSA and Q-Learning)
   2. Deep Reinforcement Learning
      - Introduction and Motivation behind
      - Value-based methods and Policy Gradient methods
      - Focus and detailed explanation of DDPG and SAC
   3. Related Work
      - Explanation of the state-of-the-art focusing more on Reda's paper, its approach and the related bibliography.
      - Increasing interest in Reinforcement Learning applied to real world situations, in contrast with simulated environment experiments.
3. Tools and Frameworks
   - Introduction to the chapter
   1. Open AI Gym
      - General description of the framework
      - Discussion about the necessity of a Reinforcement Learning Framework to test different algorithms with differents environments providing the same interface
   2. Anki Cozmo
      - General description of Cozmo and Anki
      - General informations about the mechanics and features of Cozmo
      - Discussion about the selection of Cozmo instead of other solutions
        - Amazon Deep Racer: not available at the start of the thesis. It provides a simulator to train the agent. Drawback of 
        - Building a DonkeyCar on your own: one of the best path to follow because of the personalization available. The main drawbacks is the length of the process to build the car and, above all, the time to spend in the creation of interfaces between the car and python. (FUTURE WORK)
        - In the end, Cozmo is the best choice because of provides a plain and simple control of the car and a rich Python SDK to use with OpenAI Gym.
      - Discussion about the on-board or off-board computation
   3. PyTorch
      - General Description of Pytorch framework.
      - Why Pytorch? Numpy-like framework and TensorFlow 2.0 not yet released
4. Design of the control system
   - Introduction to the chapter
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
    - 