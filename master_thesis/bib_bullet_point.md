# Bibliography Bullet Point

## Now is the Time for Reinforcement Learning on Real Robots - Alex Kendall

```tex
@misc{nowreal2019kendall,
    title={Now is the Time for Reinforcement Learning on Real Robots},
    author={Alex Kendall},
    howpublished={\url{https://alexgkendall.com/reinforcement_learning/now_is_the_time_for_reinforcement_learning_on_real_robots/}},
    year={2019}
}
```

> The more pressing problem is to be able to learn in a very data-efficient manner, retaining information and avoiding catastrophic forgetting of information. When we first applied a model-free reinforcement learning algorithm to our self-driving car, DDPG, it was very difficult to get it to learn quickly, and then to not forget stable policies. Dealing with the random-ness of random exploration made experiments incredibly stochastic and unreliable. We found model-based reinforcement learning to be more repeatable and effective. This allowed us to learn off-policy with much more deterministic results, with policies which improved with more data. Ultimately, to scale our driving policy to public UK roads, we needed to use expert demonstrations.

> For an interesting anecdote for reward design, we observed that when our car was trained with the reward to drive as far as possible without safety driver intervention, it learned to zig-zag down the road, as it didn’t leave the lane and cause intervention, but drove a greater distance by zig-zaging, therefore maximising the reward. This is a phenomenon known as reward hacking, where the agent earns reward using unintended behavior. For an excellent treatment of reward-hacking and other problems in AI safety, see Amodei et al. 2016.


> There are not many reinforcement learning systems that work with computer vision. There are also not many reinforcement learning systems that work with state-spaces with millions of dimensions. Therefore if our robots are going to work with mega-pixel camera sensors, we are going to need to learn policies that combine RL and computer vision.


> If we take the approach of, ‘let’s solve A.G.I. in simulation first and only then let’s solve it in the real-world’, we are not going to see real benefits of intelligent robotics in the near future. There is a huge opportunity to work on A.I. for robotics today. Hardware is cheaper, more accessible and reliable than ever before. I think mobile robotics is about to go through the revolution that computer vision, NLP and other data science fields have seen over the last five years.

