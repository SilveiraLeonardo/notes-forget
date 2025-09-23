Notes about optimization in sequential learning

*September 24, 2025*

### Last week results

The last thing we discussed last meeting was the somewhat confusing results obtained when testing for the *savings* of the method.

If we recall, the idea was to measure, after the model has trained on a Task A, and then on a Task B, how much faster would it learn Task A now when compared with the first time it learned it.

```
Task A -> Task B -> Task A
```

The results were complicated: they varied strongly with the optimizer used, and were not always positive, meaning the model took more time to learn the task the second time around. Secondly, for optimizers with a state, there was a significant difference in the result depending if the state was reset between tasks or not.


|     | SGD | SGD + momentum | SGD + momentum | Adam | Adam | RMSProp | RMSProp |
|------------|------- |------- |------- |------- |------- |------- |------- |
| Reset state   | - | No | Yes | No | Yes | No | Yes |
| Learning rate | 0.1 | 0.01 | 0.01 | 0.001 | 0.001 | 0.001 | 0.001 |
| Savings    | 0.59 | -0.11 | -0.09 | -0.74 | 0.13 | 0.57 | 0.70 |

** Results are the mean of 25 runs.

Adding to these, we cited two papers advising against the use of Adam in favor of vanilla SGD for sequential learning tasks (`Mirzadeh et al., 2020`, `Ashley & Sutton, 2021`). I was not convinced by their argumentation, and thought that the matter of optimization in the sequential learning scenario needed more investigation.

But one thing can be highligted before proceeding: the metric *savings* was inspired by the work from `Ebbinghaus, 1885`, in his pioneering memory studies. In its original definition, *savings* was a measure of the inner stability of what was learned - so a measure of the state of knowledge itself.

Translating this to neural networks, the stability of what was learned depends only on the knowledge encoded in the parameters of the neural network, and therefore it a measure inherent to the model itself (like accuracy would be).

But the way we are measure *savings* in neural network makes it not dependent only on the state of knowledge of the model, but also on the optimizer, and therefore it is dependent of a number of hyperparameters as well. This makes the concept of what we are measuring very confusing and hard to interpret.

***Soft/early conclusion:*** The measure of *savings* is not measuring what it says it is, that is: *the stability of what the model has learned*, and therefore biasing the interpretation of results in the field. This conclusion suggests it should no be used.

### About the state of the optimizer

A point that warrants further investigation is the different performance obtained when resetting and not resetting the optimizers state between tasks.

To test this, we abandon the notion of *savings*, to a different setup that we can deal without mixing up interpretations from psychology. The idea is, first we train the model sequentially on a number of tasks:

```
tasks = [
    [1, 2],
    [3, 4], 
    [5, 6],
    [7, 8],
    [9, 0],
]
```

Everytime the model trains on a task, it goes to the solution boundary to that task, and may forget about the task it saw before. But intuitively we believe that, as the model trains on more of this tasks, it may be easier to get to the joint solution for all the classes.

So we measure, after the sequential learning, how fast the model is able to get to the final solution, training for all taks concurrently.

```
base_task = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
```

This can be interpreted as:



![reset](./images_general/opt_reset_no_reset_2.png)






