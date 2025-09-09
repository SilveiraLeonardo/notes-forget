# Catastrophic forgetting

This file goes through my thought process on the subject, starting from the early research,
then current research, benchmarks and state-of-the-art, discussing my experiments so far
and finally the next questions I think are interesting to try to answer next.

## Early research

The problem of catastrophic forgetting (or catastrophic interference) was well research
in the late eighties and nineties. Papers such as `Richard S. Sutton, 1986`, 
`Michael McCloskey & Neal J . Cohen, 1989`, `Roger Ratcliff, 1990` and `Robert M. French, 1999` 
lay much of the fundation of what we know today, and are arguably still the best reference 
we have to understand the phenomenon.

Put it simply, catastrophic forgetting is what happens when we train a neural network to train task A,
and after it we train it to perform a second task B. The expected is the now the network is able to perform
both tasks, but what we get is a network that can perform task B and forgot how to perform task A.

The magnitude of this *forgetting* is most illustrative on the comparison with human subjects: 
`Barnes & Underwood, 1959` taught the subjects of 8 pair of random associated words, list A-B.
After the subjects had successfully learned the pairings, they were taught a second 8 pair, 
now associating the same stimuli A with a second response C, list A-C.

After learning the A-C pairs to varying degrees, the subjects were tested to see how much 
retention they still had of the A-B pairs - that is, the measure of how much *interference* 
or *forgetting* had been caused by the sequence learning protocol.

```
Teach A-B pairs of words

Teach A-C pairs of words

Test on the retention of the A-B pairs
```

As the subjects got better at the recall of the A-C list, their ability for recalling
the A-B list decreased, to about 52%. It is noted that regardless the level of the learning 
in the second list, unlearning the first list is virtually never complete and rarely exceeds 50%.

![a-b a-c experiment](./images_general/barnes_underwood.png)

`Michael McCloskey & Neal J . Cohen, 1989` replicated this experiment using a small neural network,
that would learn to map `stimulus + context` to a `response`. The stimulus represented the A list,
the response the B or C lists, and the context informed the network which list it was on.

The results were pretty remarkable: The network learned the to map the pairs A-B just fine. 
Training on the second task, the network also learn how to do the A-C mapping. But, even 
*before* the network had made any correct prediction on the A-C list, the performance on 
the A-B list had gone to zero.

![a-b a-c nn](./images_general/ab-ac-nn.png)

We replicated this experiment, and got the same result: 

![sequential acc](./image/sequential_acc.png)

In practice, what happens is that the network learns to the A-B sequence at first,
but after learning the A-C sequence it will not output B, only C, even thought each
`stimulus` is paired with a `context` vector indicating which list pair should be considered.

These early studies identified two causes for this phenomenon:

1. The nature of gradient descent optimization; and
2. The use of distributed representation.

### Gradient descent optimization

`Michael McCloskey & Neal J . Cohen, 1989` illustrated the problem of the gradient descent
learning algorithm in terms of the search in the weight space of a solution region for the 
problem at hand, that we understand as a region of low loss. If we have two tasks two learn,
we want the network to find a solution regions that is good for both. In the weight space, 
that means the intersection of the solution regions for the two problems.

When we train the network for both problems at the same time, both solutions are pulling our 
weights towards their direction, and as a consequence we move to a place that satisfies both
constraints.

Is sequential learning, however, our weights are pulled by just one solution at a time: 
We first move directly to the solution space of the first problem, and when we start training 
on the second problem our weights are pulled directly toward that direction, with nothing holding 
us back to the solution space of problem 1. We may as well finish training farther from solution 1 
than we started.

Therefore, we can say that once training for solution 1 stops and for solution 2 starts, 
there is nothing to cause the weights to go to the overall solution space.

![weight-space-1](./images_general/weight-space-1.png)

![weight-space-2](./images_general/weight-space-2.png)

`French, 1999` points that after we found the solution for our first problem, when we train 
for the second problem, even if we move just a little, it may disrupt prior learning: this 
is because the weight space is not *network friendly*, due to the presence of *weight cliffs*.

![weight cliffs](./images_general/weight-cliffs.png)

### Distributed representations

`French, 1999` and a number of other authors suggested that catastrophic forgetting was 
caused by the overlap of internal distributed representations, where changing a set of 
connections to a segment of the representation alters the output for all input examples.
Reduce this overlap could reduce catastrophic forgetting: using sparser representations 
the network could change connections targetting part of the inputs, without changing the 
output for inputs that had zeroes in those segments.

Following this reasoning, one solution for catastrophic forgetting tried for several authors 
were the use of semi-distributed representations, that kept many advantages of the fully 
distributed representations, but were not fully distributed.

However, it is also noted that reducing the overlap between different representations 
(making them sparser) also reduces the exploitation of shared structure between them - 
generalizability can only be achieved with distributed representations.


#### Catastrophic forgetting and sparse representations

This apparent inability of neural networks to learn sequentially when using distributed 
representation, and the presence of the hippocampus as a component of the human memory 
system that uses sparse representations gave inspiration to the Complementary Learning Systems 
theory ` James L. McClelland, Bruce L. McNaughton & Randall C O'Reilly, 1995`.

## Current research

The study of catastrophic forgetting was revived by `Ian J. Goodfellow et al., 2013`, 
with an empirical investigating the effect that different components of neural networks, 
such as dropout and the choice of activation, had in the forgetting effect. After that, 
*many* studies followed.

As it is usually the case, these studies vary a great deal in the protocols used for training 
and evaluation, and several claimed state-of-the-art results over the others. This unsatisfactory 
state of affairs made it very hard to compare different methods, but the work of 
`Gido M. van de Ven, Tinne Tuytelaars & Andreas S. Tolias` brought some organization:

Their main contribution was divide the sequential learning problem in three different scenarios:

1. Task incremental learning: easiest scenario, where the model is informed which task it should 
perform. It is allowed to train models with task specific components, such as a prediction head.

2. Domain-incremental learning: task identity is not available. The input distribution changes,
but the structure of the task is always the same (for instance, it has always the same number 
of output classes).

3. Class-incremental learning: task identity is not available and the structure of the task 
changes (for instance, the problem of incrementally learning new classes).

![sequential learning scenarios](./images_general/sequential-learning-scenarios.png)

Class incremental learning is most interesting and most challenging scenario.
They identified in the literature four family of methods for this problem:

1. Models that use weight regularization, to encourage important parameters for previous task 
to not change too much in later training.

2. Models that use functional regularization: using set of inputs outputs to anchor the *function* 
not too change too much. This models usually use destillation.

3. Replay based models: New training data is complemented with representative data from the past.

4. Template based models: Multi-stage multi-component methods, where first the task is inferred from the new sample,
and than the task is solved by a specialized model for that task.

![methods class incremental](./images_general/methods-class-incremental.png)

They found that, currently, the only methods capable of solving the class-incremental learning scenario 
are replay or template based.


Additionally, two benchmark datasets were used: `Split-MNIST` and `Split-CIFAR100`.






