# CIFAR 100

## Concurrent training, regular CIFAR 100

### Test 1

#### Adam

Adam with regular parameters

`def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, device="cpu")`

Epoch 0, train loss 3.4380, train acc 0.1717, val loss 3.0018, val acc 0.2375

Epoch 1, train loss 2.5202, train acc 0.3445, val loss 2.5301, val acc 0.3464

Epoch 2, train loss 2.0643, train acc 0.4419, val loss 2.1549, val acc 0.4221

Epoch 3, train loss 1.7737, train acc 0.5113, val loss 1.9711, val acc 0.4627

Epoch 4, train loss 1.5449, train acc 0.5667, val loss 1.9505, val acc 0.4757

Epoch 5, train loss 1.3612, train acc 0.6111, val loss 1.7459, val acc 0.5229

Epoch 6, train loss 1.2037, train acc 0.6547, val loss 1.6999, val acc 0.5385

Epoch 7, train loss 1.0553, train acc 0.6941, val loss 1.6258, val acc 0.5606

Epoch 8, train loss 0.9228, train acc 0.7276, val loss 1.6842, val acc 0.5584

Epoch 9, train loss 0.7980, train acc 0.7642, val loss 1.7137, val acc 0.5583

![sequential](./images_cifar/concurrently_acc_adam.png)

![sequential](./images_cifar/concurrently_loss_adam.png)

![sequential](./images_cifar/concurrently_probs_adam.png)

#### RMSProp

`def __init__(self, params, lr=1e-3, beta=0.99, eps=1e-8, device="cpu")`

Epoch 0, train loss 3.7468, train acc 0.1178, val loss 3.4092, val acc 0.1758

Epoch 1, train loss 2.8808, train acc 0.2715, val loss 2.8657, val acc 0.2751

Epoch 2, train loss 2.3749, train acc 0.3740, val loss 2.9325, val acc 0.2843

Epoch 3, train loss 2.0451, train acc 0.4484, val loss 2.5139, val acc 0.3556

Epoch 4, train loss 1.8109, train acc 0.5004, val loss 2.1570, val acc 0.4277

Epoch 5, train loss 1.6304, train acc 0.5439, val loss 2.2888, val acc 0.4316

Epoch 6, train loss 1.4677, train acc 0.5856, val loss 2.1067, val acc 0.4571

Epoch 7, train loss 1.3299, train acc 0.6168, val loss 2.1290, val acc 0.4442

Epoch 8, train loss 1.2031, train acc 0.6537, val loss 1.7893, val acc 0.5171

Epoch 9, train loss 1.0813, train acc 0.6854, val loss 2.3711, val acc 0.4427

![sequential](./images_cifar/concurrently_acc_rmsprop.png)

![sequential](./images_cifar/concurrently_loss_rmsprop.png)

#### Adadelta

Epoch 0, train loss 3.5895, train acc 0.1416, val loss 3.6502, val acc 0.1614

Epoch 1, train loss 2.6972, train acc 0.3037, val loss 2.7232, val acc 0.3034

Epoch 2, train loss 2.2076, train acc 0.4055, val loss 3.0187, val acc 0.2731

Epoch 3, train loss 1.8967, train acc 0.4791, val loss 2.1674, val acc 0.4242

Epoch 4, train loss 1.6579, train acc 0.5358, val loss 2.0045, val acc 0.4580

Epoch 5, train loss 1.4576, train acc 0.5869, val loss 2.2938, val acc 0.4201

Epoch 6, train loss 1.2842, train acc 0.6281, val loss 1.9241, val acc 0.4826

Epoch 7, train loss 1.1262, train acc 0.6713, val loss 1.9166, val acc 0.5072

Epoch 8, train loss 0.9845, train acc 0.7073, val loss 1.9346, val acc 0.5008

Epoch 9, train loss 0.8512, train acc 0.7460, val loss 1.8661, val acc 0.528

![sequential](./images_cifar/concurrently_acc_adadelta.png)

![sequential](./images_cifar/concurrently_loss_adadelta.png)

![sequential](./images_cifar/concurrently_probs_adadelta.png)

## Sequential training

Training using Adadelta, and dropout with probability 0.5

Task  0: classes = [25, 61, 42, 33, 81, 68, 6, 7, 54, 94]

Task  1: classes = [67, 71, 62, 0, 3, 41, 39, 85, 88, 16]

Task  2: classes = [82, 22, 92, 4, 14, 46, 30, 56, 28, 79]

Task  3: classes = [48, 74, 35, 24, 90, 84, 5, 95, 83, 60]

Task  4: classes = [32, 73, 47, 70, 43, 53, 20, 89, 17, 64]

Task  5: classes = [18, 76, 10, 97, 65, 44, 72, 40, 57, 78]

Task  6: classes = [77, 91, 52, 58, 93, 29, 38, 98, 37, 36]

Task  7: classes = [2, 1, 15, 23, 8, 9, 19, 51, 66, 75]

Task  8: classes = [87, 99, 63, 86, 59, 31, 50, 55, 11, 96]

Task  9: classes = [34, 13, 49, 80, 45, 26, 12, 27, 69, 21]

##### Training task 0 with classes [25, 61, 42, 33, 81, 68, 6, 7, 54, 94]

Epoch 14, train loss 0.5852, train acc 0.7968, val loss 1.4646, val acc 0.5880

On all previous tasks, loss 1.4645904331207276, acc 0.5880

![sequential](./images_cifar/sequential_acc_task1.png)

![sequential](./images_cifar/sequential_loss_task1.png)

##### Training task 1 with classes [67, 71, 62, 0, 3, 41, 39, 85, 88, 16]

Epoch 14, train loss 0.3957, train acc 0.8702, val loss 0.6616, val acc 0.8020

On all previous tasks, loss 5.5431067047119145, acc 0.401

![sequential](./images_cifar/sequential_forgetting_task2.png)

![sequential](./images_cifar/sequential_acc_task2.png)

![sequential](./images_cifar/sequential_loss_task2.png)

##### Training task 2 with classes [82, 22, 92, 4, 14, 46, 30, 56, 28, 79]

Epoch 14, train loss 0.5486, train acc 0.8100, val loss 1.2962, val acc 0.6770

On all previous tasks, loss 8.684470924377441, acc 0.22566666666666665

![sequential](./images_cifar/sequential_forgetting_task3.png)

![sequential](./images_cifar/sequential_acc_task3.png)

![sequential](./images_cifar/sequential_loss_task3.png)

##### Training task 3 with classes [48, 74, 35, 24, 90, 84, 5, 95, 83, 60]

Epoch 14, train loss 0.3682, train acc 0.8682, val loss 0.8112, val acc 0.7430

On all previous tasks, loss 8.92057144165039, acc 0.18575

##### Training task 4 with classes [32, 73, 47, 70, 43, 53, 20, 89, 17, 64]

Epoch 14, train loss 0.3220, train acc 0.8854, val loss 0.7286, val acc 0.7650

On all previous tasks, loss 9.042738537597657, acc 0.153

##### Training task 5 with classes [18, 76, 10, 97, 65, 44, 72, 40, 57, 78]

Epoch 14, train loss 0.7545, train acc 0.7444, val loss 1.2880, val acc 0.5790

On all previous tasks, loss 9.573794588724772, acc 0.0965

##### Training task 6 with classes [77, 91, 52, 58, 93, 29, 38, 98, 37, 36]

Epoch 14, train loss 0.4702, train acc 0.8374, val loss 0.8806, val acc 0.7360

On all previous tasks, loss 9.792192609514508, acc 0.10514285714285715

##### Training task 7 with classes [2, 1, 15, 23, 8, 9, 19, 51, 66, 75]

Epoch 14, train loss 0.4438, train acc 0.8488, val loss 0.8913, val acc 0.7200

On all previous tasks, loss 9.706102737426757, acc 0.09

##### Training task 8 with classes [87, 99, 63, 86, 59, 31, 50, 55, 11, 96]

Epoch 14, train loss 0.5396, train acc 0.8062, val loss 1.0123, val acc 0.7070

On all previous tasks, loss 10.512745715671116, acc 0.07855555555555556

##### Training task 9 with classes [34, 13, 49, 80, 45, 26, 12, 27, 69, 21]

Epoch 14, train loss 0.5087, train acc 0.8196, val loss 0.9262, val acc 0.7240

On all previous tasks, loss 10.658863336181641, acc 0.0724

