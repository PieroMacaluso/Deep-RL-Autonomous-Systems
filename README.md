<h1 align="center">
  <a href="https://github.com/pieromacaluso/RoomMonitor" title="RoomMonitor Documentation">
    <img alt="RoomMonitor" src="stuff/logo.svg" width="200px" height="200px" />
  </a>
  <br/>
  Deep Reinforcement Learning for autonomous systems
</h1>

<p align="center">
  Designing a control system to exploit model-free deep reinforcement learning algorithms to solve a real-world autonomous driving task of a small robot.
</p>

<p align="center">
 <img alt="Languages" src="https://img.shields.io/badge/Languages-Python-orange"/>
 <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch%20|_OpenAI_Gym%20|_Flask-green"/>
<img alt="Status" src="https://img.shields.io/badge/Status-Work In Progress-orange"/>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
  - [Candidate](#candidate)
  - [Supervisors](#supervisors)
- [Abstract](#abstract)
- [Repository Structure](#repository-structure)
  - [Master Thesis](#master-thesis)
  - [Source Code](#source-code)
    - [OpenAI Gym Environment](#openai-gym-environment)
    - [Pendulum-v0 Implementations](#pendulum-v0-implementations)
    - [CozmoDriver-v0 Implementations](#cozmodriver-v0-implementations)
  - [Reports](#reports)
- [Contibutions and License](#contibutions-and-license)
- [References](#references)
  - [Video Lectures](#video-lectures)
  - [Books](#books)
  - [Papers](#papers)
  - [Repositories](#repositories)

## Introduction

The Official repository of my master thesis in Computer Engineering at Politecnico di Torino.

The project was developed at Eurecom (Biot, France) with prof. Pietro Michiardi (Eurecom) and prof. Elena Baralis (Politecnico di Torino).

### Candidate

- <img alt="avatar" src="https://github.com/pieromacaluso.png" width="20px" height="20px"> **Piero Macaluso** - [pieromacaluso](https://github.com/pieromacaluso)

### Supervisors

- <img alt="avatar" src="https://github.com/michiard.png" width="20px" height="20px"> **Prof. Pietro Michiardi** - [michiard](https://github.com/michiard)
- <img alt="avatar" src="https://dbdmg.polito.it/wordpress/wp-content/uploads/2010/12/Elena_tessera-150x150.jpg" width="20px" height="20px"> **Prof. Elena Baralis** - [elena.baralis](https://dbdmg.polito.it/wordpress/people/elena-baralis/)

## Abstract

Because of its potential to thoroughly change mobility and transport, autonomous systems and self-driving vehicles are attracting much attention from both the research community and industry.
Recent work has demonstrated that it is possible to rely on a comprehensive understanding of the immediate environment while following simple high-level directions, to obtain a more scalable approach that can make autonomous driving a ubiquitous technology.
However, to date, the majority of the methods concentrates on deterministic control optimisation algorithms to select the right action, while the usage of deep learning and machine learning is entirely dedicated to object detection and recognition.

Recently, we have witnessed a remarkable increase in interest in Reinforcement Learning (RL). It is a machine learning field focused on solving Markov Decision Processes (MDP), where an agent learns to make decisions by mapping situations and actions according to the information it gathers from the surrounding environment and from the reward it receives, trying to maximise it.
As researchers discovered, it can be surprisingly useful to solve tasks in simulated environments like games and computer games, and it showed encouraging performance in tasks with robotic manipulators. Furthermore, the great fervour produced by the widespread exploitation of deep learning opened the doors to function approximation with convolutional neural networks, developing what is nowadays known as deep reinforcement learning.

In this thesis, we argue that the generality of reinforcement learning makes it a useful framework where to apply autonomous driving to inject artificial intelligence not only in the detection component but also in the decision-making one.
The focus of the majority of reinforcement learning projects is on a simulated environment. However, a more challenging approach of reinforcement learning consists of the application of this type of algorithms in the real world.
For this reason, we designed and implemented a control system for Cozmo, a small toy robot developed by Anki company, by exploiting the Cozmo SDK, PyTorch and OpenAI Gym to build up a standardised environment in which to apply any reinforcement learning algorithm: it represents the first contribution of our thesis.

Furthermore, we designed a circuit where we were able to carry out experiments in the real world, the second contribution of our work.
We started from a simplified environment where to test algorithm functionalities to motivate and discuss our implementation choices.
Therefore, we implemented our version of Soft Actor-Critic (SAC), a model-free reinforcement learning algorithm suitable for real-world experiments, to solve the specific self-driving task with Cozmo. The agent managed to reach a maximum value of above 3.5 meters in the testing phase, which equals more than one complete tour of the track. Despite this significant result, it was not able to learn how to drive securely and stably. Thus, we focused on the analysis of the strengths and weaknesses of this approach outlining what could be the next steps to make this cutting-edge technology concrete and efficient.

## Repository Structure

### Master Thesis

- Master Thesis: [LaTeX Project](master_thesis) | [PDF](https://raw.githubusercontent.com/pieromacaluso/Deep-RL-Autonomous-Systems/master_thesis/master_thesis.pdf)
- Summary [LaTex](summary) | [PDF](https://raw.githubusercontent.com/pieromacaluso/Deep-RL-Autonomous-Systems/summary/summary.pdf)
- Presentation: [LaTex](presentation) | [PDF](https://raw.githubusercontent.com/pieromacaluso/Deep-RL-Autonomous-Systems/presentation/presentation.pdf)

### Source Code

#### OpenAI Gym Environment

- [Gym-Cozmo](gym-cozmo)

#### Pendulum-v0 Implementations

- [DDPG NN](source_code/NN_DDPG_implementation)
- [DDPG CNN](source_code/CNN_DDPG_implementation)
- [SAC CNN](source_code/SAC_implementation)

#### CozmoDriver-v0 Implementations

- [DDPG](source_code/ddpg_cozmo)
- [SAC](source_code/sac_cozmo)

### Reports

- April 2019: Deep Deterministic Policy Gradient (DDPG) - [REPORT](https://github.com/pieromacaluso/Deep-RL-Autonomous-Systems/raw/master/reports/report_DDPG/report_DDPG.pdf)
- May 2019: Soft Actor-Critic (SAC) - [REPORT](https://github.com/pieromacaluso/Deep-RL-Autonomous-Systems/raw/master/reports/report_SAC/report_SAC.pdf)
- September 2019: Experiment Flow - [REPORT](https://github.com/pieromacaluso/Deep-RL-Autonomous-Systems/raw/master/reports/report_09092019/report_09092019.pdf)

## Contibutions and License

It is possible to fork the project and create your own one following the rules given by the [LICENSE](LICENSE).

Please cite using the following BibTex entry:

```latex
@mastersthesis{macaluso2020deep,
  author  = {Piero Macaluso},
  title   = {{Deep Reinforcement Learning for Autonomous Systems}},
  school  = {{Politecnico di Torino}, {Eurecom}}
  year    = {2020}
}
```

If you want to contribute or to request a new features, you can do that via the ISSUE sections.

If you need any help to setup the project or to have information about it, feel free to join us at <a href="https://t.me/PieroMacaluso">`@PieroMacaluso` on Telegram</a> and ask away.

## References
  
### Video Lectures

**[v1]** [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

**[v2]** [Reinforcement Learning Udacity Course](https://classroom.udacity.com/courses/ud600)

### Books

**[b1]** [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2018.pdf) (2018) by Richard S. Sutton and Andrew G. Barto

**[b2]** [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands) (2018) by Maxim Lapan

### Papers

**[p1]** [Learning to Drive in a Day](https://arxiv.org/pdf/1807.00412.pdf) (Sep 2018) by Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley & Amar Shah

**[p2]** [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) (Feb 2016) by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra

**[p3]** [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) (2014) David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller

**[p4]** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290) (2018) Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine

**[p5]** [Soft Actor-Critic Algorithms and Applications](http://proceedings.mlr.press/v32/silver14.pdf) (2018) Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine

### Repositories

- [Exercises about Silver's videolectures](https://github.com/dennybritz/reinforcement-learning)
