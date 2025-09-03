# Notes - Retrospective inference experiment

## Concurrent or interleaved learning

As expected, the model was able to get 100% accuracy on the 16 patterns.

Epoch 0, loss 0.6894, acc 0.0000

Epoch 1, loss 0.7599, acc 0.0000

Epoch 2, loss 0.6436, acc 0.0000

Epoch 3, loss 0.5864, acc 0.0000

Epoch 4, loss 0.5352, acc 0.0625

Epoch 5, loss 0.4961, acc 0.1250

Epoch 6, loss 0.4634, acc 0.0625

...

Epoch 30, loss 0.0437, acc 0.8750

Epoch 31, loss 0.0399, acc 0.8750

Epoch 32, loss 0.0354, acc 0.8750

Epoch 33, loss 0.0315, acc 0.9375

Epoch 34, loss 0.0273, acc 1.0000

![concurrent loss](./image/concurrent_loss.png)
![concurrent acc](./image/concurrent_acc.png)

## Sequentially learning

Now training first the 8 patterns from the AB list, and then the 8 pattersn from the AC list.

The first 8 patterns are learned fairly quicky, in less than 15 iterations. The second pattern takes much longer to achieve 100%, around 40 epochs. The previous learning seems to have made more difficult for the next learning to occur.

Also it is very interesting to see the evolution of the accuracy lines for the previous and current tasks:

The accuracy for the current task only start increasing after the accuracy of the previous task when to zero. And this happens very quickly, in only 6 epochs.

This make be linked to the fact that the new learning took more epochs to occur: one may think that this is because the network had first to forget what it had learned before.

![sequential acc](./image/sequential_acc.png)
