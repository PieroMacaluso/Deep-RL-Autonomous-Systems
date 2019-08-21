# Latest Ideas

## Learning phase 21/08/19
After increasing the number of epochs to 1000, we noticed a strange effect on CNN, the weights and bias changes only a little, or not at all.
We think that this fact is due to the size of replay memory buffer, which remains almost the same from one episode to the next one, leading the network to sample and learn from the same pool of memory.
For this reason, we decide to enter in the learning phase only with a difference of dimension of `batch_size` from the previous learning step. We are testing if this modification will improve the learning process.