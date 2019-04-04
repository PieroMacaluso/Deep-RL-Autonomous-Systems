# A Study Of Reinforcement Learning
The official report of my Master Thesis in Computer Engineering at Politecnico di Torino.
The project was developed at Eurecom (Biot, France) with prof. Pietro Michiardi (Eurecom) and prof. Elena Baralis (Politecnico di Torino) from March 2019 to TBD.
## Path of my Master Thesis
- Studying the fundamentals of Reinforcement Learning throught watching **[v1](#video-lectures)**, reading some chapters from **[b1](#books)** and doing some personal research.
- Studying **[p1](#papers)** and **[p2](#papers)** in order to understand the algorithm to reproduce.
- Producing an initial working code of DDPG algorithm on a continuous and well-known environment on the OpenAI Gym environment.
  - **Main aim:** try to implement the algorithm as generic as possible in order to mantain flexibility
  - **Possible developments**: Find out if it could be better to develop the code with **[D4PG (Deep Distributed Distributional Deterministic Policy Gradients) ](https://arxiv.org/pdf/1804.08617.pdf)** or **[SAC (Soft Actor-Critic) ](https://arxiv.org/pdf/1801.01290.pdf)**
- Generalizing the code to work with CNN and images in an environment as similar as possible to the target one ([Anki Cozmo](https://www.anki.com/en-us/cozmo))
- Implement the code on Anki Cozmo creating an interface between it and OpenAI Gym
- ... ***TBD*** ...
 
  
## Video Lectures
**[v1]** [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

**[v2]** [Reinforcement Learning Udacity Course](https://classroom.udacity.com/courses/ud600)

## Books
**[b1]** [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) (2018) by Richard S. Sutton and Andrew G. Barto

**[b2]** [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands) (2018) by Maxim Lapan

## Papers
**[p1]** [Learning to Drive in a Day](https://arxiv.org/pdf/1807.00412.pdf) (Sep 2018) by Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley & Amar Shah

**[p2]** [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) (Feb 2016) by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra

**[p3]** [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) (2014) David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller



## Repositories
- [Exercises about Silver's videolectures](https://github.com/dennybritz/reinforcement-learning)
