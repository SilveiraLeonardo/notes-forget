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

![sequential](./images_cifar/concurrently_latent.png)

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

### Longer training

Training using Adadelta, and dropout with probability 0.5, for 200 epochs (or 90\% acc on validation set, whatever comes first):

**Task 1**

Epoch 199, train loss 0.0485, train acc 0.9828, val loss 1.0216, val acc 0.8110

On all previous tasks, loss 1.0216185398101807, acc 0.811

Accuracy    | Task 1 |
|------------|------- |
| Classifier                     | 0.8400 |
| Class                couch    | 0.8000 |
| Class                plate    | 0.9474 |
| Class              leopard    | 0.7727 |
| Class               forest    | 0.8947 |
| Class            streetcar    | 0.8462 |
| Class                 road    | 0.8333 |
| Class                  bee    | 0.7917 |
| Class               beetle    | 0.7778 |
| Class               orchid    | 0.8750 |
| Class             wardrobe    | 0.9231 |

**Task 2**

Epoch 118, train loss 0.0488, train acc 0.9840, val loss 0.5626, val acc 0.9020

On all previous tasks, loss 6.09733037185669, acc 0.451

| Accuracy    | Task 1 | Task 2 |
|------------|------- |------- |
| Classifier                     | 0.8400 | 0.7650 |
| Class                couch    | 0.8000 | 0.6296 |
| Class                plate    | 0.9474 | 0.8000 |
| Class              leopard    | 0.7727 | 0.6000 |
| Class               forest    | 0.8947 | 0.4118 |
| Class            streetcar    | 0.8462 | 0.9091 |
| Class                 road    | 0.8333 | 0.7778 |
| Class                  bee    | 0.7917 | 0.6667 |
| Class               beetle    | 0.7778 | 0.6000 |
| Class               orchid    | 0.8750 | 0.8000 |
| Class             wardrobe    | 0.9231 | 0.8500 |
| Class                  ray    |        | 0.7333 |
| Class                  sea    |        | 0.8750 |
| Class                poppy    |        | 0.8966 |
| Class                apple    |        | 0.9565 |
| Class                 bear    |        | 0.6154 |
| Class           lawn_mower    |        | 0.8889 |
| Class             keyboard    |        | 0.8000 |
| Class                 tank    |        | 0.8800 |
| Class                tiger    |        | 0.7917 |
| Class                  can    |        | 0.6000 |

**Task 3**

Epoch 199, train loss 0.0591, train acc 0.9790, val loss 0.8057, val acc 0.8480

On all previous tasks, loss 8.354194271087646, acc 0.283

| Accuracy    | Task 1 | Task 2 | Task 3 |
|------------|------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 |
| Class                  ray    |        | 0.7333 | 0.8125 |
| Class                  sea    |        | 0.8750 | 0.9000 |
| Class                poppy    |        | 0.8966 | 0.5455 |
| Class                apple    |        | 0.9565 | 0.8095 |
| Class                 bear    |        | 0.6154 | 0.6500 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 |
| Class             keyboard    |        | 0.8000 | 0.7778 |
| Class                 tank    |        | 0.8800 | 1.0000 |
| Class                tiger    |        | 0.7917 | 0.5909 |
| Class                  can    |        | 0.6000 | 0.4800 |
| Class            sunflower    |        |        | 0.8824 |
| Class                clock    |        |        | 0.4000 |
| Class                tulip    |        |        | 0.4762 |
| Class               beaver    |        |        | 0.6818 |
| Class            butterfly    |        |        | 0.4737 |
| Class                  man    |        |        | 0.7000 |
| Class              dolphin    |        |        | 0.6875 |
| Class            palm_tree    |        |        | 0.9048 |
| Class                  cup    |        |        | 0.6316 |
| Class               spider    |        |        | 0.6333 |

**Task 4**

Epoch 75, train loss 0.0818, train acc 0.9710, val loss 0.4257, val acc 0.9030

On all previous tasks, loss 9.039847373962402, acc 0.23825

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 |
|------------|------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 |
| Class                clock    |        |        | 0.4000 | 0.5789 |
| Class                tulip    |        |        | 0.4762 | 0.1250 |
| Class               beaver    |        |        | 0.6818 | 0.3500 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 |
| Class                  man    |        |        | 0.7000 | 0.4800 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 |
| Class                  cup    |        |        | 0.6316 | 0.7647 |
| Class               spider    |        |        | 0.6333 | 0.6190 |
| Class           motorcycle    |        |        |        | 0.8636 |
| Class                shrew    |        |        |        | 0.4737 |
| Class                 girl    |        |        |        | 0.6842 |
| Class            cockroach    |        |        |        | 0.8750 |
| Class                train    |        |        |        | 0.4231 |
| Class                table    |        |        |        | 0.6000 |
| Class                  bed    |        |        |        | 0.8333 |
| Class                whale    |        |        |        | 0.4118 |
| Class         sweet_pepper    |        |        |        | 0.4706 |
| Class                plain    |        |        |        | 0.6818 |

**Task 5**

Epoch 154, train loss 0.0415, train acc 0.9854, val loss 0.4745, val acc 0.9020

On all previous tasks, loss 9.023781416320801, acc 0.182

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|------------|------- |------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 | 0.5600 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 | 0.1500 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 | 0.8667 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 | 0.5172 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 | 0.5263 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 | 0.7500 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 | 0.7059 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 | 0.5000 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 | 0.5000 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 | 0.5385 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 | 0.7143 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 | 0.3889 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 | 0.7333 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 | 0.4444 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 | 0.6667 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 | 0.2500 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 | 0.5000 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 | 0.5417 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 | 0.5600 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 | 0.5714 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 | 0.4545 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 | 0.7917 |
| Class                clock    |        |        | 0.4000 | 0.5789 | 0.4000 |
| Class                tulip    |        |        | 0.4762 | 0.1250 | 0.5714 |
| Class               beaver    |        |        | 0.6818 | 0.3500 | 0.2222 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 | 0.3684 |
| Class                  man    |        |        | 0.7000 | 0.4800 | 0.6500 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 | 0.5238 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 | 0.8571 |
| Class                  cup    |        |        | 0.6316 | 0.7647 | 0.8125 |
| Class               spider    |        |        | 0.6333 | 0.6190 | 0.5455 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.7037 |
| Class                shrew    |        |        |        | 0.4737 | 0.3750 |
| Class                 girl    |        |        |        | 0.6842 | 0.3158 |
| Class            cockroach    |        |        |        | 0.8750 | 0.7059 |
| Class                train    |        |        |        | 0.4231 | 0.6111 |
| Class                table    |        |        |        | 0.6000 | 0.6087 |
| Class                  bed    |        |        |        | 0.8333 | 0.2632 |
| Class                whale    |        |        |        | 0.4118 | 0.6818 |
| Class         sweet_pepper    |        |        |        | 0.4706 | 0.2727 |
| Class                plain    |        |        |        | 0.6818 | 0.7000 |
| Class             flatfish    |        |        |        |        | 0.5238 |
| Class                shark    |        |        |        |        | 0.4615 |
| Class           maple_tree    |        |        |        |        | 0.8947 |
| Class                 rose    |        |        |        |        | 0.4500 |
| Class                 lion    |        |        |        |        | 0.5333 |
| Class               orange    |        |        |        |        | 0.6800 |
| Class                chair    |        |        |        |        | 0.7895 |
| Class              tractor    |        |        |        |        | 0.5556 |
| Class               castle    |        |        |        |        | 0.8000 |
| Class               possum    |        |        |        |        | 0.4839 |

**Task 6**

Epoch 199, train loss 0.0824, train acc 0.9706, val loss 1.0874, val acc 0.7580

On all previous tasks, loss 9.926072171529134, acc 0.12783333333333333

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
|------------|------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 | 0.5600 | 0.4758 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 | 0.1500 | 0.3571 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 | 0.8667 | 0.7222 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 | 0.5172 | 0.4167 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 | 0.5263 | 0.5882 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 | 0.7500 | 0.6923 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 | 0.7059 | 0.6522 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 | 0.5000 | 0.3158 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 | 0.5000 | 0.2667 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 | 0.5385 | 0.4783 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 | 0.7143 | 0.8571 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 | 0.3889 | 0.4444 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 | 0.7333 | 0.5455 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 | 0.4444 | 0.3810 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 | 0.6667 | 0.6667 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 | 0.2500 | 0.3571 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 | 0.5000 | 0.6667 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 | 0.5417 | 0.5000 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 | 0.5600 | 0.4800 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 | 0.5714 | 0.6667 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 | 0.4545 | 0.5263 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 | 0.7917 | 0.7333 |
| Class                clock    |        |        | 0.4000 | 0.5789 | 0.4000 | 0.1481 |
| Class                tulip    |        |        | 0.4762 | 0.1250 | 0.5714 | 0.3913 |
| Class               beaver    |        |        | 0.6818 | 0.3500 | 0.2222 | 0.3200 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 | 0.3684 | 0.2727 |
| Class                  man    |        |        | 0.7000 | 0.4800 | 0.6500 | 0.4286 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 | 0.5238 | 0.2381 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 | 0.8571 | 0.6190 |
| Class                  cup    |        |        | 0.6316 | 0.7647 | 0.8125 | 0.7500 |
| Class               spider    |        |        | 0.6333 | 0.6190 | 0.5455 | 0.4000 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.7037 | 0.5600 |
| Class                shrew    |        |        |        | 0.4737 | 0.3750 | 0.3333 |
| Class                 girl    |        |        |        | 0.6842 | 0.3158 | 0.2727 |
| Class            cockroach    |        |        |        | 0.8750 | 0.7059 | 0.6667 |
| Class                train    |        |        |        | 0.4231 | 0.6111 | 0.2500 |
| Class                table    |        |        |        | 0.6000 | 0.6087 | 0.2593 |
| Class                  bed    |        |        |        | 0.8333 | 0.2632 | 0.2857 |
| Class                whale    |        |        |        | 0.4118 | 0.6818 | 0.6875 |
| Class         sweet_pepper    |        |        |        | 0.4706 | 0.2727 | 0.2857 |
| Class                plain    |        |        |        | 0.6818 | 0.7000 | 0.6087 |
| Class             flatfish    |        |        |        |        | 0.5238 | 0.3500 |
| Class                shark    |        |        |        |        | 0.4615 | 0.5217 |
| Class           maple_tree    |        |        |        |        | 0.8947 | 0.7826 |
| Class                 rose    |        |        |        |        | 0.4500 | 0.3448 |
| Class                 lion    |        |        |        |        | 0.5333 | 0.4483 |
| Class               orange    |        |        |        |        | 0.6800 | 0.5000 |
| Class                chair    |        |        |        |        | 0.7895 | 0.6923 |
| Class              tractor    |        |        |        |        | 0.5556 | 0.5652 |
| Class               castle    |        |        |        |        | 0.8000 | 0.7500 |
| Class               possum    |        |        |        |        | 0.4839 | 0.3684 |
| Class          caterpillar    |        |        |        |        |        | 0.4444 |
| Class           skyscraper    |        |        |        |        |        | 0.8824 |
| Class                 bowl    |        |        |        |        |        | 0.4706 |
| Class                 wolf    |        |        |        |        |        | 0.5200 |
| Class               rabbit    |        |        |        |        |        | 0.2941 |
| Class               lizard    |        |        |        |        |        | 0.2308 |
| Class                 seal    |        |        |        |        |        | 0.3684 |
| Class                 lamp    |        |        |        |        |        | 0.5000 |
| Class                 pear    |        |        |        |        |        | 0.4583 |
| Class                snake    |        |        |        |        |        | 0.3158 |

**Task 7**

Epoch 199, train loss 0.0555, train acc 0.9796, val loss 0.5873, val acc 0.8650

On all previous tasks, loss 10.052490291050502, acc 0.12414285714285714

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 |
|------------|------- |------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 | 0.5600 | 0.4758 | 0.4514 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 | 0.1500 | 0.3571 | 0.5000 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 | 0.8667 | 0.7222 | 0.4286 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 | 0.5172 | 0.4167 | 0.4615 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 | 0.5263 | 0.5882 | 0.4800 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 | 0.7500 | 0.6923 | 0.5500 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 | 0.7059 | 0.6522 | 0.6087 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 | 0.5000 | 0.3158 | 0.4000 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 | 0.5000 | 0.2667 | 0.3684 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 | 0.5385 | 0.4783 | 0.7778 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 | 0.7143 | 0.8571 | 0.7895 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 | 0.3889 | 0.4444 | 0.3636 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 | 0.7333 | 0.5455 | 0.6667 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 | 0.4444 | 0.3810 | 0.2400 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 | 0.6667 | 0.6667 | 0.6316 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 | 0.2500 | 0.3571 | 0.3200 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 | 0.5000 | 0.6667 | 0.6522 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 | 0.5417 | 0.5000 | 0.3548 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 | 0.5600 | 0.4800 | 0.5200 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 | 0.5714 | 0.6667 | 0.4000 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 | 0.4545 | 0.5263 | 0.6842 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 | 0.7917 | 0.7333 | 0.6471 |
| Class                clock    |        |        | 0.4000 | 0.5789 | 0.4000 | 0.1481 | 0.3333 |
| Class                tulip    |        |        | 0.4762 | 0.1250 | 0.5714 | 0.3913 | 0.2353 |
| Class               beaver    |        |        | 0.6818 | 0.3500 | 0.2222 | 0.3200 | 0.1875 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 | 0.3684 | 0.2727 | 0.3103 |
| Class                  man    |        |        | 0.7000 | 0.4800 | 0.6500 | 0.4286 | 0.4286 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 | 0.5238 | 0.2381 | 0.3500 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 | 0.8571 | 0.6190 | 0.7895 |
| Class                  cup    |        |        | 0.6316 | 0.7647 | 0.8125 | 0.7500 | 0.6250 |
| Class               spider    |        |        | 0.6333 | 0.6190 | 0.5455 | 0.4000 | 0.2941 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.7037 | 0.5600 | 0.9048 |
| Class                shrew    |        |        |        | 0.4737 | 0.3750 | 0.3333 | 0.3043 |
| Class                 girl    |        |        |        | 0.6842 | 0.3158 | 0.2727 | 0.2308 |
| Class            cockroach    |        |        |        | 0.8750 | 0.7059 | 0.6667 | 0.7273 |
| Class                train    |        |        |        | 0.4231 | 0.6111 | 0.2500 | 0.3913 |
| Class                table    |        |        |        | 0.6000 | 0.6087 | 0.2593 | 0.3000 |
| Class                  bed    |        |        |        | 0.8333 | 0.2632 | 0.2857 | 0.3636 |
| Class                whale    |        |        |        | 0.4118 | 0.6818 | 0.6875 | 0.3636 |
| Class         sweet_pepper    |        |        |        | 0.4706 | 0.2727 | 0.2857 | 0.3636 |
| Class                plain    |        |        |        | 0.6818 | 0.7000 | 0.6087 | 0.7368 |
| Class             flatfish    |        |        |        |        | 0.5238 | 0.3500 | 0.3529 |
| Class                shark    |        |        |        |        | 0.4615 | 0.5217 | 0.3600 |
| Class           maple_tree    |        |        |        |        | 0.8947 | 0.7826 | 0.3889 |
| Class                 rose    |        |        |        |        | 0.4500 | 0.3448 | 0.4783 |
| Class                 lion    |        |        |        |        | 0.5333 | 0.4483 | 0.2800 |
| Class               orange    |        |        |        |        | 0.6800 | 0.5000 | 0.6667 |
| Class                chair    |        |        |        |        | 0.7895 | 0.6923 | 0.5000 |
| Class              tractor    |        |        |        |        | 0.5556 | 0.5652 | 0.4783 |
| Class               castle    |        |        |        |        | 0.8000 | 0.7500 | 0.6842 |
| Class               possum    |        |        |        |        | 0.4839 | 0.3684 | 0.2727 |
| Class          caterpillar    |        |        |        |        |        | 0.4444 | 0.2727 |
| Class           skyscraper    |        |        |        |        |        | 0.8824 | 0.7917 |
| Class                 bowl    |        |        |        |        |        | 0.4706 | 0.3750 |
| Class                 wolf    |        |        |        |        |        | 0.5200 | 0.4091 |
| Class               rabbit    |        |        |        |        |        | 0.2941 | 0.2500 |
| Class               lizard    |        |        |        |        |        | 0.2308 | 0.1250 |
| Class                 seal    |        |        |        |        |        | 0.3684 | 0.2000 |
| Class                 lamp    |        |        |        |        |        | 0.5000 | 0.3077 |
| Class                 pear    |        |        |        |        |        | 0.4583 | 0.4737 |
| Class                snake    |        |        |        |        |        | 0.3158 | 0.2632 |
| Class                snail    |        |        |        |        |        |        | 0.3684 |
| Class                trout    |        |        |        |        |        |        | 0.6667 |
| Class             oak_tree    |        |        |        |        |        |        | 0.8571 |
| Class         pickup_truck    |        |        |        |        |        |        | 0.4000 |
| Class               turtle    |        |        |        |        |        |        | 0.0625 |
| Class             dinosaur    |        |        |        |        |        |        | 0.5000 |
| Class             kangaroo    |        |        |        |        |        |        | 0.5625 |
| Class                woman    |        |        |        |        |        |        | 0.3571 |
| Class                house    |        |        |        |        |        |        | 0.7143 |
| Class              hamster    |        |        |        |        |        |        | 0.7222 |

**Task 8**

Epoch 199, train loss 0.0456, train acc 0.9866, val loss 0.5422, val acc 0.8710

On all previous tasks, loss 10.181735054016114, acc 0.11275

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 |
|------------|------- |------- |------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 | 0.5600 | 0.4758 | 0.4514 | 0.4444 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 | 0.1500 | 0.3571 | 0.5000 | 0.2000 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 | 0.8667 | 0.7222 | 0.4286 | 0.3810 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 | 0.5172 | 0.4167 | 0.4615 | 0.6111 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 | 0.5263 | 0.5882 | 0.4800 | 0.3125 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 | 0.7500 | 0.6923 | 0.5500 | 0.4000 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 | 0.7059 | 0.6522 | 0.6087 | 0.8421 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 | 0.5000 | 0.3158 | 0.4000 | 0.3684 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 | 0.5000 | 0.2667 | 0.3684 | 0.2222 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 | 0.5385 | 0.4783 | 0.7778 | 0.4545 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 | 0.7143 | 0.8571 | 0.7895 | 0.8500 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 | 0.3889 | 0.4444 | 0.3636 | 0.3333 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 | 0.7333 | 0.5455 | 0.6667 | 0.7500 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 | 0.4444 | 0.3810 | 0.2400 | 0.2500 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 | 0.6667 | 0.6667 | 0.6316 | 0.6400 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 | 0.2500 | 0.3571 | 0.3200 | 0.2917 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 | 0.5000 | 0.6667 | 0.6522 | 0.5500 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 | 0.5417 | 0.5000 | 0.3548 | 0.3913 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 | 0.5600 | 0.4800 | 0.5200 | 0.5217 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 | 0.5714 | 0.6667 | 0.4000 | 0.3000 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 | 0.4545 | 0.5263 | 0.6842 | 0.6667 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 | 0.7917 | 0.7333 | 0.6471 | 0.4583 |
| Class                clock    |        |        | 0.4000 | 0.5789 | 0.4000 | 0.1481 | 0.3333 | 0.3333 |
| Class                tulip    |        |        | 0.4762 | 0.1250 | 0.5714 | 0.3913 | 0.2353 | 0.2143 |
| Class               beaver    |        |        | 0.6818 | 0.3500 | 0.2222 | 0.3200 | 0.1875 | 0.1739 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 | 0.3684 | 0.2727 | 0.3103 | 0.2400 |
| Class                  man    |        |        | 0.7000 | 0.4800 | 0.6500 | 0.4286 | 0.4286 | 0.3043 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 | 0.5238 | 0.2381 | 0.3500 | 0.5000 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 | 0.8571 | 0.6190 | 0.7895 | 0.5882 |
| Class                  cup    |        |        | 0.6316 | 0.7647 | 0.8125 | 0.7500 | 0.6250 | 0.4737 |
| Class               spider    |        |        | 0.6333 | 0.6190 | 0.5455 | 0.4000 | 0.2941 | 0.5238 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.7037 | 0.5600 | 0.9048 | 0.8889 |
| Class                shrew    |        |        |        | 0.4737 | 0.3750 | 0.3333 | 0.3043 | 0.0588 |
| Class                 girl    |        |        |        | 0.6842 | 0.3158 | 0.2727 | 0.2308 | 0.2000 |
| Class            cockroach    |        |        |        | 0.8750 | 0.7059 | 0.6667 | 0.7273 | 0.3333 |
| Class                train    |        |        |        | 0.4231 | 0.6111 | 0.2500 | 0.3913 | 0.3529 |
| Class                table    |        |        |        | 0.6000 | 0.6087 | 0.2593 | 0.3000 | 0.2273 |
| Class                  bed    |        |        |        | 0.8333 | 0.2632 | 0.2857 | 0.3636 | 0.3529 |
| Class                whale    |        |        |        | 0.4118 | 0.6818 | 0.6875 | 0.3636 | 0.3636 |
| Class         sweet_pepper    |        |        |        | 0.4706 | 0.2727 | 0.2857 | 0.3636 | 0.2500 |
| Class                plain    |        |        |        | 0.6818 | 0.7000 | 0.6087 | 0.7368 | 0.7500 |
| Class             flatfish    |        |        |        |        | 0.5238 | 0.3500 | 0.3529 | 0.2069 |
| Class                shark    |        |        |        |        | 0.4615 | 0.5217 | 0.3600 | 0.3478 |
| Class           maple_tree    |        |        |        |        | 0.8947 | 0.7826 | 0.3889 | 0.4545 |
| Class                 rose    |        |        |        |        | 0.4500 | 0.3448 | 0.4783 | 0.5000 |
| Class                 lion    |        |        |        |        | 0.5333 | 0.4483 | 0.2800 | 0.5625 |
| Class               orange    |        |        |        |        | 0.6800 | 0.5000 | 0.6667 | 0.7857 |
| Class                chair    |        |        |        |        | 0.7895 | 0.6923 | 0.5000 | 0.6667 |
| Class              tractor    |        |        |        |        | 0.5556 | 0.5652 | 0.4783 | 0.4211 |
| Class               castle    |        |        |        |        | 0.8000 | 0.7500 | 0.6842 | 0.5500 |
| Class               possum    |        |        |        |        | 0.4839 | 0.3684 | 0.2727 | 0.1579 |
| Class          caterpillar    |        |        |        |        |        | 0.4444 | 0.2727 | 0.3000 |
| Class           skyscraper    |        |        |        |        |        | 0.8824 | 0.7917 | 0.7895 |
| Class                 bowl    |        |        |        |        |        | 0.4706 | 0.3750 | 0.2667 |
| Class                 wolf    |        |        |        |        |        | 0.5200 | 0.4091 | 0.1905 |
| Class               rabbit    |        |        |        |        |        | 0.2941 | 0.2500 | 0.2857 |
| Class               lizard    |        |        |        |        |        | 0.2308 | 0.1250 | 0.2500 |
| Class                 seal    |        |        |        |        |        | 0.3684 | 0.2000 | 0.2353 |
| Class                 lamp    |        |        |        |        |        | 0.5000 | 0.3077 | 0.4583 |
| Class                 pear    |        |        |        |        |        | 0.4583 | 0.4737 | 0.4167 |
| Class                snake    |        |        |        |        |        | 0.3158 | 0.2632 | 0.3333 |
| Class                snail    |        |        |        |        |        |        | 0.3684 | 0.2941 |
| Class                trout    |        |        |        |        |        |        | 0.6667 | 0.5789 |
| Class             oak_tree    |        |        |        |        |        |        | 0.8571 | 0.5000 |
| Class         pickup_truck    |        |        |        |        |        |        | 0.4000 | 0.6364 |
| Class               turtle    |        |        |        |        |        |        | 0.0625 | 0.4615 |
| Class             dinosaur    |        |        |        |        |        |        | 0.5000 | 0.4000 |
| Class             kangaroo    |        |        |        |        |        |        | 0.5625 | 0.3889 |
| Class                woman    |        |        |        |        |        |        | 0.3571 | 0.0909 |
| Class                house    |        |        |        |        |        |        | 0.7143 | 0.6111 |
| Class              hamster    |        |        |        |        |        |        | 0.7222 | 0.6000 |
| Class                 baby    |        |        |        |        |        |        |        | 0.3600 |
| Class        aquarium_fish    |        |        |        |        |        |        |        | 0.7391 |
| Class                camel    |        |        |        |        |        |        |        | 0.5000 |
| Class                cloud    |        |        |        |        |        |        |        | 0.6786 |
| Class              bicycle    |        |        |        |        |        |        |        | 0.7619 |
| Class               bottle    |        |        |        |        |        |        |        | 0.6000 |
| Class               cattle    |        |        |        |        |        |        |        | 0.5714 |
| Class             mushroom    |        |        |        |        |        |        |        | 0.4375 |
| Class              raccoon    |        |        |        |        |        |        |        | 0.6667 |
| Class                skunk    |        |        |        |        |        |        |        | 0.9091 |

**Task 9**

Epoch 199, train loss 0.0643, train acc 0.9776, val loss 0.7775, val acc 0.8180

On all previous tasks, loss 10.389605711195204, acc 0.09211111111111112

**Task 10**

Epoch 199, train loss 0.0602, train acc 0.9790, val loss 0.6840, val acc 0.8500

On all previous tasks, loss 10.669502571105957, acc 0.0859

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
|------------|------- |------- |------- |------- |------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.8400 | 0.7650 | 0.6683 | 0.5837 | 0.5600 | 0.4758 | 0.4514 | 0.4444 | 0.3861 | 0.3835 |
| Class                couch    | 0.8000 | 0.6296 | 0.4783 | 0.3636 | 0.1500 | 0.3571 | 0.5000 | 0.2000 | 0.2500 | 0.1250 |
| Class                plate    | 0.9474 | 0.8000 | 0.7619 | 0.5625 | 0.8667 | 0.7222 | 0.4286 | 0.3810 | 0.5238 | 0.2667 |
| Class              leopard    | 0.7727 | 0.6000 | 0.4286 | 0.6400 | 0.5172 | 0.4167 | 0.4615 | 0.6111 | 0.5000 | 0.5000 |
| Class               forest    | 0.8947 | 0.4118 | 0.6000 | 0.7895 | 0.5263 | 0.5882 | 0.4800 | 0.3125 | 0.4375 | 0.6250 |
| Class            streetcar    | 0.8462 | 0.9091 | 0.8148 | 0.4444 | 0.7500 | 0.6923 | 0.5500 | 0.4000 | 0.4783 | 0.4667 |
| Class                 road    | 0.8333 | 0.7778 | 0.9444 | 0.7826 | 0.7059 | 0.6522 | 0.6087 | 0.8421 | 0.5263 | 0.5357 |
| Class                  bee    | 0.7917 | 0.6667 | 0.3889 | 0.5217 | 0.5000 | 0.3158 | 0.4000 | 0.3684 | 0.2727 | 0.2667 |
| Class               beetle    | 0.7778 | 0.6000 | 0.5862 | 0.3478 | 0.5000 | 0.2667 | 0.3684 | 0.2222 | 0.3684 | 0.4000 |
| Class               orchid    | 0.8750 | 0.8000 | 0.7273 | 0.4500 | 0.5385 | 0.4783 | 0.7778 | 0.4545 | 0.6000 | 0.5200 |
| Class             wardrobe    | 0.9231 | 0.8500 | 0.9286 | 0.8750 | 0.7143 | 0.8571 | 0.7895 | 0.8500 | 0.6667 | 0.6818 |
| Class                  ray    |        | 0.7333 | 0.8125 | 0.5000 | 0.3889 | 0.4444 | 0.3636 | 0.3333 | 0.2353 | 0.4706 |
| Class                  sea    |        | 0.8750 | 0.9000 | 0.7368 | 0.7333 | 0.5455 | 0.6667 | 0.7500 | 0.6087 | 0.4762 |
| Class                poppy    |        | 0.8966 | 0.5455 | 0.5600 | 0.4444 | 0.3810 | 0.2400 | 0.2500 | 0.6154 | 0.4167 |
| Class                apple    |        | 0.9565 | 0.8095 | 0.6842 | 0.6667 | 0.6667 | 0.6316 | 0.6400 | 0.6111 | 0.6250 |
| Class                 bear    |        | 0.6154 | 0.6500 | 0.4667 | 0.2500 | 0.3571 | 0.3200 | 0.2917 | 0.2632 | 0.1200 |
| Class           lawn_mower    |        | 0.8889 | 0.6667 | 0.7500 | 0.5000 | 0.6667 | 0.6522 | 0.5500 | 0.6500 | 0.6842 |
| Class             keyboard    |        | 0.8000 | 0.7778 | 0.4286 | 0.5417 | 0.5000 | 0.3548 | 0.3913 | 0.3500 | 0.5000 |
| Class                 tank    |        | 0.8800 | 1.0000 | 0.7500 | 0.5600 | 0.4800 | 0.5200 | 0.5217 | 0.2941 | 0.5417 |
| Class                tiger    |        | 0.7917 | 0.5909 | 0.5600 | 0.5714 | 0.6667 | 0.4000 | 0.3000 | 0.3500 | 0.4667 |
| Class                  can    |        | 0.6000 | 0.4800 | 0.5000 | 0.4545 | 0.5263 | 0.6842 | 0.6667 | 0.5909 | 0.5000 |
| Class            sunflower    |        |        | 0.8824 | 0.6667 | 0.7917 | 0.7333 | 0.6471 | 0.4583 | 0.6000 | 0.5000 |
| Class                clock    |        |        | 0.4000 | 0.5789 | 0.4000 | 0.1481 | 0.3333 | 0.3333 | 0.1538 | 0.3125 |
| Class                tulip    |        |        | 0.4762 | 0.1250 | 0.5714 | 0.3913 | 0.2353 | 0.2143 | 0.3500 | 0.1818 |
| Class               beaver    |        |        | 0.6818 | 0.3500 | 0.2222 | 0.3200 | 0.1875 | 0.1739 | 0.2188 | 0.1600 |
| Class            butterfly    |        |        | 0.4737 | 0.5833 | 0.3684 | 0.2727 | 0.3103 | 0.2400 | 0.1429 | 0.1111 |
| Class                  man    |        |        | 0.7000 | 0.4800 | 0.6500 | 0.4286 | 0.4286 | 0.3043 | 0.2308 | 0.2857 |
| Class              dolphin    |        |        | 0.6875 | 0.5455 | 0.5238 | 0.2381 | 0.3500 | 0.5000 | 0.3636 | 0.1600 |
| Class            palm_tree    |        |        | 0.9048 | 0.8947 | 0.8571 | 0.6190 | 0.7895 | 0.5882 | 0.7727 | 0.6190 |
| Class                  cup    |        |        | 0.6316 | 0.7647 | 0.8125 | 0.7500 | 0.6250 | 0.4737 | 0.3684 | 0.6667 |
| Class               spider    |        |        | 0.6333 | 0.6190 | 0.5455 | 0.4000 | 0.2941 | 0.5238 | 0.4000 | 0.3333 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.7037 | 0.5600 | 0.9048 | 0.8889 | 0.6154 | 0.6522 |
| Class                shrew    |        |        |        | 0.4737 | 0.3750 | 0.3333 | 0.3043 | 0.0588 | 0.2727 | 0.1579 |
| Class                 girl    |        |        |        | 0.6842 | 0.3158 | 0.2727 | 0.2308 | 0.2000 | 0.1852 | 0.1304 |
| Class            cockroach    |        |        |        | 0.8750 | 0.7059 | 0.6667 | 0.7273 | 0.3333 | 0.5789 | 0.3684 |
| Class                train    |        |        |        | 0.4231 | 0.6111 | 0.2500 | 0.3913 | 0.3529 | 0.2174 | 0.2222 |
| Class                table    |        |        |        | 0.6000 | 0.6087 | 0.2593 | 0.3000 | 0.2273 | 0.2857 | 0.2000 |
| Class                  bed    |        |        |        | 0.8333 | 0.2632 | 0.2857 | 0.3636 | 0.3529 | 0.3182 | 0.4118 |
| Class                whale    |        |        |        | 0.4118 | 0.6818 | 0.6875 | 0.3636 | 0.3636 | 0.4286 | 0.3077 |
| Class         sweet_pepper    |        |        |        | 0.4706 | 0.2727 | 0.2857 | 0.3636 | 0.2500 | 0.3684 | 0.2800 |
| Class                plain    |        |        |        | 0.6818 | 0.7000 | 0.6087 | 0.7368 | 0.7500 | 0.7273 | 0.6897 |
| Class             flatfish    |        |        |        |        | 0.5238 | 0.3500 | 0.3529 | 0.2069 | 0.2000 | 0.1739 |
| Class                shark    |        |        |        |        | 0.4615 | 0.5217 | 0.3600 | 0.3478 | 0.1667 | 0.3077 |
| Class           maple_tree    |        |        |        |        | 0.8947 | 0.7826 | 0.3889 | 0.4545 | 0.2857 | 0.6111 |
| Class                 rose    |        |        |        |        | 0.4500 | 0.3448 | 0.4783 | 0.5000 | 0.3889 | 0.2500 |
| Class                 lion    |        |        |        |        | 0.5333 | 0.4483 | 0.2800 | 0.5625 | 0.3000 | 0.2273 |
| Class               orange    |        |        |        |        | 0.6800 | 0.5000 | 0.6667 | 0.7857 | 0.5000 | 0.7368 |
| Class                chair    |        |        |        |        | 0.7895 | 0.6923 | 0.5000 | 0.6667 | 0.6250 | 0.3684 |
| Class              tractor    |        |        |        |        | 0.5556 | 0.5652 | 0.4783 | 0.4211 | 0.3846 | 0.5882 |
| Class               castle    |        |        |        |        | 0.8000 | 0.7500 | 0.6842 | 0.5500 | 0.6316 | 0.7143 |
| Class               possum    |        |        |        |        | 0.4839 | 0.3684 | 0.2727 | 0.1579 | 0.1250 | 0.2632 |
| Class          caterpillar    |        |        |        |        |        | 0.4444 | 0.2727 | 0.3000 | 0.5000 | 0.3529 |
| Class           skyscraper    |        |        |        |        |        | 0.8824 | 0.7917 | 0.7895 | 0.5000 | 0.5882 |
| Class                 bowl    |        |        |        |        |        | 0.4706 | 0.3750 | 0.2667 | 0.1538 | 0.2143 |
| Class                 wolf    |        |        |        |        |        | 0.5200 | 0.4091 | 0.1905 | 0.2941 | 0.3158 |
| Class               rabbit    |        |        |        |        |        | 0.2941 | 0.2500 | 0.2857 | 0.0909 | 0.1364 |
| Class               lizard    |        |        |        |        |        | 0.2308 | 0.1250 | 0.2500 | 0.1538 | 0.0714 |
| Class                 seal    |        |        |        |        |        | 0.3684 | 0.2000 | 0.2353 | 0.1500 | 0.0000 |
| Class                 lamp    |        |        |        |        |        | 0.5000 | 0.3077 | 0.4583 | 0.3500 | 0.2692 |
| Class                 pear    |        |        |        |        |        | 0.4583 | 0.4737 | 0.4167 | 0.2917 | 0.5238 |
| Class                snake    |        |        |        |        |        | 0.3158 | 0.2632 | 0.3333 | 0.2000 | 0.2308 |
| Class                snail    |        |        |        |        |        |        | 0.3684 | 0.2941 | 0.2727 | 0.1923 |
| Class                trout    |        |        |        |        |        |        | 0.6667 | 0.5789 | 0.4762 | 0.3000 |
| Class             oak_tree    |        |        |        |        |        |        | 0.8571 | 0.5000 | 0.4375 | 0.4667 |
| Class         pickup_truck    |        |        |        |        |        |        | 0.4000 | 0.6364 | 0.4444 | 0.1875 |
| Class               turtle    |        |        |        |        |        |        | 0.0625 | 0.4615 | 0.3000 | 0.1304 |
| Class             dinosaur    |        |        |        |        |        |        | 0.5000 | 0.4000 | 0.4167 | 0.4118 |
| Class             kangaroo    |        |        |        |        |        |        | 0.5625 | 0.3889 | 0.2353 | 0.2273 |
| Class                woman    |        |        |        |        |        |        | 0.3571 | 0.0909 | 0.1053 | 0.2000 |
| Class                house    |        |        |        |        |        |        | 0.7143 | 0.6111 | 0.3125 | 0.4000 |
| Class              hamster    |        |        |        |        |        |        | 0.7222 | 0.6000 | 0.5000 | 0.5217 |
| Class                 baby    |        |        |        |        |        |        |        | 0.3600 | 0.2105 | 0.2632 |
| Class        aquarium_fish    |        |        |        |        |        |        |        | 0.7391 | 0.4000 | 0.3333 |
| Class                camel    |        |        |        |        |        |        |        | 0.5000 | 0.3182 | 0.2174 |
| Class                cloud    |        |        |        |        |        |        |        | 0.6786 | 0.5882 | 0.6111 |
| Class              bicycle    |        |        |        |        |        |        |        | 0.7619 | 0.7143 | 0.7368 |
| Class               bottle    |        |        |        |        |        |        |        | 0.6000 | 0.4444 | 0.5000 |
| Class               cattle    |        |        |        |        |        |        |        | 0.5714 | 0.3333 | 0.4737 |
| Class             mushroom    |        |        |        |        |        |        |        | 0.4375 | 0.3333 | 0.4737 |
| Class              raccoon    |        |        |        |        |        |        |        | 0.6667 | 0.4211 | 0.3889 |
| Class                skunk    |        |        |        |        |        |        |        | 0.9091 | 0.5000 | 0.5000 |
| Class           television    |        |        |        |        |        |        |        |        | 0.5217 | 0.8000 |
| Class                 worm    |        |        |        |        |        |        |        |        | 0.4286 | 0.2500 |
| Class            porcupine    |        |        |        |        |        |        |        |        | 0.4118 | 0.2500 |
| Class            telephone    |        |        |        |        |        |        |        |        | 0.7895 | 0.3600 |
| Class            pine_tree    |        |        |        |        |        |        |        |        | 0.4167 | 0.3571 |
| Class             elephant    |        |        |        |        |        |        |        |        | 0.6842 | 0.3636 |
| Class                mouse    |        |        |        |        |        |        |        |        | 0.1500 | 0.1429 |
| Class                otter    |        |        |        |        |        |        |        |        | 0.1667 | 0.0000 |
| Class                  boy    |        |        |        |        |        |        |        |        | 0.2222 | 0.3333 |
| Class          willow_tree    |        |        |        |        |        |        |        |        | 0.3684 | 0.2917 |
| Class                  fox    |        |        |        |        |        |        |        |        |        | 0.4286 |
| Class                  bus    |        |        |        |        |        |        |        |        |        | 0.1500 |
| Class             mountain    |        |        |        |        |        |        |        |        |        | 0.6364 |
| Class             squirrel    |        |        |        |        |        |        |        |        |        | 0.2500 |
| Class              lobster    |        |        |        |        |        |        |        |        |        | 0.3684 |
| Class                 crab    |        |        |        |        |        |        |        |        |        | 0.6667 |
| Class               bridge    |        |        |        |        |        |        |        |        |        | 0.5217 |
| Class            crocodile    |        |        |        |        |        |        |        |        |        | 0.4667 |
| Class               rocket    |        |        |        |        |        |        |        |        |        | 0.7647 |
| Class           chimpanzee    |        |        |        |        |        |        |        |        |        | 0.7273 |

#### How general are the features learned?

How well a linear classifier can separate all classes, after training on each of the ten tasks?

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
|------------|------- |------- |------- |------- |------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.3225 | 0.3335 | 0.3490 | 0.3520 | 0.3580 | 0.3330 | 0.3710 | 0.3810 | 0.3840 | 0.3700 |
| Class                couch    | 0.2083 | 0.3333 | 0.1667 | 0.3750 | 0.2917 | 0.2083 | 0.2500 | 0.2083 | 0.2500 | 0.3333 |
| Class                plate    | 0.2667 | 0.3333 | 0.3333 | 0.4000 | 0.2667 | 0.4000 | 0.3333 | 0.2667 | 0.2667 | 0.2000 |
| Class              leopard    | 0.5000 | 0.4444 | 0.2778 | 0.4444 | 0.4444 | 0.2222 | 0.3889 | 0.3889 | 0.5000 | 0.2778 |
| Class               forest    | 0.6875 | 0.5625 | 0.2500 | 0.5625 | 0.3750 | 0.3125 | 0.3750 | 0.4375 | 0.3750 | 0.7500 |
| Class            streetcar    | 0.4667 | 0.4000 | 0.4667 | 0.4667 | 0.4000 | 0.2000 | 0.2667 | 0.2000 | 0.2667 | 0.3333 |
| Class                 road    | 0.6786 | 0.6071 | 0.6071 | 0.6429 | 0.6071 | 0.5357 | 0.5714 | 0.7857 | 0.5714 | 0.5714 |
| Class                  bee    | 0.6000 | 0.6000 | 0.5333 | 0.4667 | 0.4000 | 0.2000 | 0.4667 | 0.4667 | 0.2000 | 0.2667 |
| Class               beetle    | 0.6000 | 0.3000 | 0.3000 | 0.3000 | 0.5000 | 0.3000 | 0.2000 | 0.2000 | 0.6000 | 0.5000 |
| Class               orchid    | 0.6000 | 0.6000 | 0.6000 | 0.4000 | 0.5200 | 0.4800 | 0.4400 | 0.4800 | 0.4800 | 0.3200 |
| Class             wardrobe    | 0.8182 | 0.6818 | 0.6364 | 0.7727 | 0.6364 | 0.5000 | 0.6818 | 0.6818 | 0.7273 | 0.7727 |
| Class                  ray    | 0.2353 | 0.2353 | 0.4118 | 0.2941 | 0.2941 | 0.3529 | 0.1765 | 0.4118 | 0.1765 | 0.2941 |
| Class                  sea    | 0.5714 | 0.6667 | 0.5238 | 0.5714 | 0.5238 | 0.7143 | 0.5238 | 0.6190 | 0.4762 | 0.5714 |
| Class                poppy    | 0.2500 | 0.4583 | 0.3333 | 0.1667 | 0.2917 | 0.3333 | 0.4167 | 0.3333 | 0.3333 | 0.2917 |
| Class                apple    | 0.7188 | 0.7500 | 0.7500 | 0.7188 | 0.6875 | 0.6562 | 0.7500 | 0.7500 | 0.7500 | 0.6562 |
| Class                 bear    | 0.1600 | 0.1600 | 0.0800 | 0.1600 | 0.1600 | 0.2400 | 0.2000 | 0.1200 | 0.2800 | 0.2400 |
| Class           lawn_mower    | 0.6316 | 0.8421 | 0.6842 | 0.6316 | 0.6842 | 0.7368 | 0.6842 | 0.4211 | 0.6842 | 0.6316 |
| Class             keyboard    | 0.4375 | 0.6875 | 0.5000 | 0.4375 | 0.4375 | 0.3750 | 0.3750 | 0.3125 | 0.4375 | 0.3125 |
| Class                 tank    | 0.5000 | 0.5833 | 0.6667 | 0.2917 | 0.6250 | 0.5417 | 0.5000 | 0.5833 | 0.6250 | 0.5000 |
| Class                tiger    | 0.3333 | 0.5333 | 0.3333 | 0.5333 | 0.4000 | 0.2000 | 0.2667 | 0.2667 | 0.5333 | 0.2667 |
| Class                  can    | 0.2500 | 0.5000 | 0.5000 | 0.4500 | 0.4000 | 0.5000 | 0.4500 | 0.3000 | 0.5500 | 0.4000 |
| Class            sunflower    | 0.6111 | 0.5000 | 0.7222 | 0.5000 | 0.6111 | 0.3889 | 0.4444 | 0.5556 | 0.5000 | 0.4444 |
| Class                clock    | 0.2500 | 0.4375 | 0.4375 | 0.3750 | 0.2500 | 0.2500 | 0.3125 | 0.3125 | 0.3125 | 0.3125 |
| Class                tulip    | 0.2727 | 0.2273 | 0.4545 | 0.2727 | 0.3182 | 0.3182 | 0.1364 | 0.3182 | 0.2727 | 0.1818 |
| Class               beaver    | 0.0000 | 0.2000 | 0.2800 | 0.1600 | 0.2000 | 0.0800 | 0.2000 | 0.1200 | 0.2400 | 0.1200 |
| Class            butterfly    | 0.2222 | 0.1667 | 0.3889 | 0.2222 | 0.3333 | 0.1667 | 0.2778 | 0.1667 | 0.1111 | 0.1111 |
| Class                  man    | 0.1429 | 0.1905 | 0.3333 | 0.3333 | 0.2857 | 0.1905 | 0.3810 | 0.2381 | 0.4286 | 0.3333 |
| Class              dolphin    | 0.3600 | 0.2400 | 0.3600 | 0.4000 | 0.3200 | 0.2400 | 0.4400 | 0.3600 | 0.2400 | 0.2800 |
| Class            palm_tree    | 0.4286 | 0.3810 | 0.7143 | 0.5714 | 0.6667 | 0.5714 | 0.6667 | 0.5238 | 0.6190 | 0.5238 |
| Class                  cup    | 0.5333 | 0.6000 | 0.7333 | 0.7333 | 0.7333 | 0.6667 | 0.7333 | 0.6000 | 0.6667 | 0.6667 |
| Class               spider    | 0.5000 | 0.3333 | 0.6667 | 0.5000 | 0.3333 | 0.4444 | 0.2778 | 0.4444 | 0.3889 | 0.3889 |
| Class           motorcycle    | 0.3913 | 0.4783 | 0.3913 | 0.5652 | 0.6087 | 0.6522 | 0.5652 | 0.6087 | 0.6522 | 0.6087 |
| Class                shrew    | 0.0000 | 0.0526 | 0.1053 | 0.2105 | 0.1579 | 0.2105 | 0.1053 | 0.0526 | 0.1053 | 0.1579 |
| Class                 girl    | 0.0870 | 0.1304 | 0.1304 | 0.2174 | 0.1304 | 0.2174 | 0.2609 | 0.2174 | 0.0435 | 0.2174 |
| Class            cockroach    | 0.5789 | 0.5789 | 0.4211 | 0.5263 | 0.4211 | 0.3684 | 0.6316 | 0.5263 | 0.3684 | 0.6316 |
| Class                train    | 0.1111 | 0.1667 | 0.2222 | 0.5000 | 0.1667 | 0.2778 | 0.3889 | 0.1111 | 0.1667 | 0.2778 |
| Class                table    | 0.0667 | 0.2667 | 0.4000 | 0.3333 | 0.2667 | 0.2667 | 0.4000 | 0.4000 | 0.1333 | 0.2667 |
| Class                  bed    | 0.4118 | 0.2941 | 0.2941 | 0.5882 | 0.5294 | 0.4706 | 0.4706 | 0.2941 | 0.1765 | 0.1765 |
| Class                whale    | 0.3077 | 0.3462 | 0.3846 | 0.4231 | 0.3462 | 0.3077 | 0.3077 | 0.3846 | 0.3846 | 0.2308 |
| Class         sweet_pepper    | 0.3200 | 0.3600 | 0.2400 | 0.4000 | 0.4000 | 0.2400 | 0.3600 | 0.4400 | 0.3600 | 0.3600 |
| Class                plain    | 0.8276 | 0.6552 | 0.6897 | 0.6552 | 0.6552 | 0.5862 | 0.7241 | 0.7586 | 0.7241 | 0.5172 |
| Class             flatfish    | 0.1304 | 0.0870 | 0.0870 | 0.1739 | 0.2609 | 0.2174 | 0.1739 | 0.2174 | 0.1739 | 0.3478 |
| Class                shark    | 0.4615 | 0.2308 | 0.5385 | 0.4615 | 0.4615 | 0.4615 | 0.3846 | 0.3846 | 0.3846 | 0.3077 |
| Class           maple_tree    | 0.5556 | 0.3889 | 0.4444 | 0.3889 | 0.4444 | 0.3889 | 0.5000 | 0.5556 | 0.4444 | 0.3889 |
| Class                 rose    | 0.3333 | 0.3333 | 0.4167 | 0.3333 | 0.4167 | 0.3333 | 0.3333 | 0.3333 | 0.4167 | 0.4583 |
| Class                 lion    | 0.3636 | 0.4091 | 0.2727 | 0.2727 | 0.5455 | 0.3636 | 0.4091 | 0.2273 | 0.3182 | 0.3182 |
| Class               orange    | 0.6842 | 0.4211 | 0.7368 | 0.7368 | 0.8421 | 0.6842 | 0.7895 | 0.7368 | 0.8421 | 0.7895 |
| Class                chair    | 0.4737 | 0.4211 | 0.5263 | 0.4211 | 0.4737 | 0.5789 | 0.5789 | 0.6316 | 0.4211 | 0.4211 |
| Class              tractor    | 0.5294 | 0.5882 | 0.6471 | 0.5882 | 0.6471 | 0.5882 | 0.7059 | 0.4706 | 0.7059 | 0.6471 |
| Class               castle    | 0.4286 | 0.5238 | 0.5238 | 0.4762 | 0.5714 | 0.3810 | 0.4762 | 0.5238 | 0.5714 | 0.5238 |
| Class               possum    | 0.0526 | 0.1053 | 0.1053 | 0.2105 | 0.2632 | 0.2105 | 0.1579 | 0.2105 | 0.1579 | 0.1579 |
| Class          caterpillar    | 0.1765 | 0.2941 | 0.2353 | 0.1765 | 0.2941 | 0.3529 | 0.1176 | 0.2941 | 0.2353 | 0.2941 |
| Class           skyscraper    | 0.5294 | 0.7059 | 0.7059 | 0.7059 | 0.6471 | 0.7647 | 0.7647 | 0.6471 | 0.7059 | 0.6471 |
| Class                 bowl    | 0.0714 | 0.3571 | 0.2143 | 0.2143 | 0.4286 | 0.3571 | 0.4286 | 0.2143 | 0.2857 | 0.3571 |
| Class                 wolf    | 0.3684 | 0.3158 | 0.3158 | 0.1053 | 0.2105 | 0.3684 | 0.3158 | 0.3684 | 0.4211 | 0.2632 |
| Class               rabbit    | 0.0909 | 0.0455 | 0.2727 | 0.2273 | 0.1818 | 0.1818 | 0.2273 | 0.1364 | 0.0909 | 0.0455 |
| Class               lizard    | 0.0714 | 0.0714 | 0.0714 | 0.1429 | 0.2857 | 0.2143 | 0.1429 | 0.0714 | 0.2857 | 0.1429 |
| Class                 seal    | 0.0500 | 0.0000 | 0.1500 | 0.0500 | 0.1500 | 0.1000 | 0.1500 | 0.1000 | 0.2000 | 0.0500 |
| Class                 lamp    | 0.3077 | 0.2308 | 0.2308 | 0.2692 | 0.2308 | 0.3077 | 0.2692 | 0.1923 | 0.2692 | 0.3077 |
| Class                 pear    | 0.0952 | 0.2381 | 0.3810 | 0.2857 | 0.4286 | 0.5714 | 0.3810 | 0.3333 | 0.3810 | 0.5238 |
| Class                snake    | 0.1923 | 0.0769 | 0.1923 | 0.0769 | 0.1538 | 0.2308 | 0.1923 | 0.2308 | 0.1538 | 0.2308 |
| Class                snail    | 0.0385 | 0.1538 | 0.1154 | 0.1154 | 0.1538 | 0.2308 | 0.3846 | 0.3462 | 0.1923 | 0.3077 |
| Class                trout    | 0.2000 | 0.2500 | 0.3000 | 0.2500 | 0.2000 | 0.2000 | 0.4000 | 0.4000 | 0.2500 | 0.2000 |
| Class             oak_tree    | 0.4000 | 0.6000 | 0.5333 | 0.4000 | 0.3333 | 0.2667 | 0.7333 | 0.5333 | 0.3333 | 0.4667 |
| Class         pickup_truck    | 0.3125 | 0.2500 | 0.3750 | 0.3750 | 0.2500 | 0.1875 | 0.3750 | 0.3125 | 0.3750 | 0.1875 |
| Class               turtle    | 0.1739 | 0.0000 | 0.0435 | 0.0435 | 0.0870 | 0.1304 | 0.1304 | 0.0435 | 0.0870 | 0.0870 |
| Class             dinosaur    | 0.4118 | 0.2353 | 0.4118 | 0.2941 | 0.3529 | 0.2941 | 0.3529 | 0.4706 | 0.4706 | 0.3529 |
| Class             kangaroo    | 0.1364 | 0.1364 | 0.0455 | 0.0455 | 0.1818 | 0.1364 | 0.1364 | 0.1818 | 0.0455 | 0.2273 |
| Class                woman    | 0.0500 | 0.1000 | 0.2000 | 0.2000 | 0.0500 | 0.0500 | 0.1500 | 0.1000 | 0.2000 | 0.2500 |
| Class                house    | 0.5000 | 0.3500 | 0.3500 | 0.3000 | 0.4000 | 0.2000 | 0.4500 | 0.3500 | 0.4000 | 0.4500 |
| Class              hamster    | 0.2609 | 0.1739 | 0.2174 | 0.2609 | 0.3478 | 0.2609 | 0.4783 | 0.4783 | 0.4783 | 0.3913 |
| Class                 baby    | 0.1053 | 0.2105 | 0.0526 | 0.2105 | 0.1579 | 0.1053 | 0.1579 | 0.3684 | 0.3158 | 0.2105 |
| Class        aquarium_fish    | 0.2963 | 0.2593 | 0.3333 | 0.2963 | 0.3704 | 0.4074 | 0.2963 | 0.4815 | 0.4444 | 0.3333 |
| Class                camel    | 0.3043 | 0.0870 | 0.2609 | 0.1304 | 0.2174 | 0.0435 | 0.1304 | 0.3913 | 0.3043 | 0.3913 |
| Class                cloud    | 0.6111 | 0.5556 | 0.6111 | 0.7778 | 0.6111 | 0.4444 | 0.3889 | 0.6111 | 0.6667 | 0.6667 |
| Class              bicycle    | 0.5263 | 0.6316 | 0.6842 | 0.6842 | 0.6316 | 0.5263 | 0.5263 | 0.8421 | 0.7895 | 0.3684 |
| Class               bottle    | 0.7143 | 0.6429 | 0.4286 | 0.5714 | 0.3571 | 0.4286 | 0.5714 | 0.5000 | 0.4286 | 0.6429 |
| Class               cattle    | 0.3684 | 0.2632 | 0.2632 | 0.2105 | 0.3684 | 0.2105 | 0.3158 | 0.4737 | 0.4737 | 0.2632 |
| Class             mushroom    | 0.2105 | 0.2632 | 0.1579 | 0.3158 | 0.3158 | 0.2632 | 0.3684 | 0.5789 | 0.3684 | 0.2105 |
| Class              raccoon    | 0.1111 | 0.1111 | 0.2222 | 0.1667 | 0.1111 | 0.1111 | 0.2778 | 0.3889 | 0.3333 | 0.2778 |
| Class                skunk    | 0.2727 | 0.3636 | 0.2727 | 0.4091 | 0.4545 | 0.3636 | 0.4545 | 0.7273 | 0.5909 | 0.5909 |
| Class           television    | 0.5000 | 0.5000 | 0.6000 | 0.5500 | 0.5500 | 0.5000 | 0.4500 | 0.5000 | 0.6500 | 0.6500 |
| Class                 worm    | 0.1250 | 0.0417 | 0.2500 | 0.1667 | 0.0417 | 0.1667 | 0.0000 | 0.0833 | 0.1667 | 0.2917 |
| Class            porcupine    | 0.2083 | 0.1667 | 0.1667 | 0.3333 | 0.1667 | 0.2500 | 0.2917 | 0.2083 | 0.3750 | 0.4167 |
| Class            telephone    | 0.2800 | 0.3200 | 0.4000 | 0.3600 | 0.3600 | 0.2800 | 0.2400 | 0.4400 | 0.4400 | 0.4000 |
| Class            pine_tree    | 0.3571 | 0.4286 | 0.2857 | 0.5000 | 0.4286 | 0.2857 | 0.4286 | 0.5000 | 0.7857 | 0.4286 |
| Class             elephant    | 0.1364 | 0.2727 | 0.2273 | 0.2727 | 0.1364 | 0.1364 | 0.2273 | 0.2273 | 0.3182 | 0.2273 |
| Class                mouse    | 0.0000 | 0.0714 | 0.0000 | 0.0000 | 0.0714 | 0.0714 | 0.0714 | 0.1429 | 0.3571 | 0.2857 |
| Class                otter    | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1538 | 0.0000 | 0.1538 | 0.0769 | 0.0769 |
| Class                  boy    | 0.0370 | 0.0741 | 0.1852 | 0.1111 | 0.1481 | 0.1111 | 0.1852 | 0.2963 | 0.3704 | 0.1852 |
| Class          willow_tree    | 0.2917 | 0.2917 | 0.2083 | 0.2917 | 0.3750 | 0.4167 | 0.3750 | 0.2917 | 0.3750 | 0.3750 |
| Class                  fox    | 0.0000 | 0.2857 | 0.1429 | 0.0000 | 0.2143 | 0.2857 | 0.5000 | 0.4286 | 0.1429 | 0.5000 |
| Class                  bus    | 0.2500 | 0.4000 | 0.3500 | 0.3500 | 0.3500 | 0.2000 | 0.3000 | 0.3000 | 0.3000 | 0.4000 |
| Class             mountain    | 0.3636 | 0.3182 | 0.0909 | 0.3182 | 0.3182 | 0.3636 | 0.4091 | 0.4091 | 0.4545 | 0.4545 |
| Class             squirrel    | 0.0417 | 0.0833 | 0.0417 | 0.0833 | 0.0417 | 0.2917 | 0.1250 | 0.0833 | 0.1667 | 0.0417 |
| Class              lobster    | 0.1053 | 0.1053 | 0.1579 | 0.2632 | 0.2632 | 0.1579 | 0.2105 | 0.1579 | 0.3158 | 0.3158 |
| Class                 crab    | 0.4167 | 0.1667 | 0.1667 | 0.1667 | 0.3333 | 0.4167 | 0.4167 | 0.5833 | 0.2500 | 0.3333 |
| Class               bridge    | 0.3478 | 0.4783 | 0.3043 | 0.4348 | 0.4783 | 0.4348 | 0.4783 | 0.3913 | 0.3913 | 0.6087 |
| Class            crocodile    | 0.2667 | 0.2000 | 0.2667 | 0.3333 | 0.2000 | 0.2000 | 0.2667 | 0.4000 | 0.3333 | 0.3333 |
| Class               rocket    | 0.4118 | 0.4118 | 0.3529 | 0.2941 | 0.2353 | 0.4706 | 0.5294 | 0.5294 | 0.5294 | 0.5882 |
| Class           chimpanzee    | 0.3182 | 0.5000 | 0.4091 | 0.5455 | 0.5000 | 0.5000 | 0.5909 | 0.5909 | 0.4545 | 0.6818 |

##### Without batch-norm:

Task 1:

Epoch 99, train loss 0.3283, train acc 0.8986, val loss 0.6666, val acc 0.8120

On all previous tasks, loss 0.6666326169967651, acc 0.812

Norm of the classification rows for the current classes and mean value:

[0.4175397  0.56702656 0.89548856 0.6914094  0.6988364  0.77886087
 0.36308578 0.503378   0.7164211  0.7393091 ]

0.6371355

Sparcity analysis - population sparcity: 0.8464

Linear probe overall acc: 0.8100

Task 2:

Epoch 99, train loss 0.1635, train acc 0.9566, val loss 0.5831, val acc 0.8600

On all previous tasks, loss 7.350429832458496, acc 0.43

Norm of the classification rows for the current classes and mean value:

[0.7140545  1.0298038  1.0241181  0.6717928  0.47595683 0.3708613
 0.83809745 0.8244912  0.86480767 0.8034659 ]

0.7617449

Norm of the classification rows for the previously seen classes and mean value:
[0.5882545  0.7579095  1.0869232  0.8061225  0.7833535  0.9386863
 0.47935894 0.70601326 0.86865747 0.98291963]

0.7998199

Sparcity analysis - population sparcity: 0.8533

Linear probe overall acc: 0.6100

Task 3:

Epoch 99, train loss 0.1955, train acc 0.9384, val loss 0.8335, val acc 0.7870

On all previous tasks, loss 8.490302618662517, acc 0.2643333333333333

Norm of the classification rows for the current classes and mean value:

[1.3686795  0.96428347 1.5770549  1.2764666  1.5847256  0.5296163
 1.100167   1.0540218  1.3936213  1.6089005 ]
1.2457536

Norm of the classification rows for the previously seen classes and mean value:

[0.75806004 0.92965454 1.3803011  0.9502555  0.8692953  1.1982458
 0.7623709  0.85161835 1.0322847  1.320072   0.86822844 1.588892
 1.1671853  0.903136   0.99767196 0.4414222  1.0403265  1.0551811
 1.3548877  1.0246707 ]

1.024688

Sparcity analysis - population sparcity: 0.8615


Linear probe overall acc: 0.4750

Task 4:

Epoch 99, train loss 0.1867, train acc 0.9470, val loss 0.5062, val acc 0.8560

On all previous tasks, loss 9.865634315490723, acc 0.21775

Norm of the classification rows for the current classes and mean value:

[1.2151967  1.6304742  1.4741209  0.8212238  1.5126164  1.0280843
 1.4336593  0.92851424 1.6389484  0.41927138]

1.210211

Norm of the classification rows for the previously seen classes and mean value:

[1.0188358  1.2854311  1.537746   1.16167    1.0234411  1.4795343
 1.0006046  1.0059994  1.3517116  1.762713   1.060818   1.8780758
 1.3835037  1.17108    1.2656143  0.71811265 1.2674688  1.3970233
 1.4982662  1.7118366  1.5284402  1.2104607  2.2395234  1.750586
 1.734439   0.6549303  1.2732158  1.2088052  1.6190295  1.9308791 ]

1.3709931

Sparcity analysis - population sparcity: 0.8761

Linear probe overall acc: 0.4238

Task 5:

Epoch 99, train loss 0.1873, train acc 0.9424, val loss 0.6315, val acc 0.8470

On all previous tasks, loss 9.273758543395996, acc 0.1722

Norm of the classification rows for the current classes and mean value:

[3.0117407 1.9510472 2.74846   0.9309722 0.7406464 2.001261  1.7498534
 2.810449  1.673527  1.3138086]
1.8931764

Norm of the classification rows for the previously seen classes and mean value:

[1.2492387  1.4793041  1.7036707  1.3835305  1.2230222  2.1903923
 1.442517   1.412112   1.6201856  2.285006   1.3157336  2.0306
 2.0518026  1.4719566  1.4764148  0.9203698  1.8883628  2.0269573
 1.6257279  2.0331132  1.6127555  1.5619359  2.771722   2.1920907
 1.9942616  0.7229828  1.504994   1.3452783  1.9717515  2.344121
 1.6552839  1.7420381  1.864071   3.526768   1.8619543  1.1998868
 1.7066306  1.1605012  2.0714028  0.42383853]

1.7016071

Sparcity analysis - population sparcity: 0.9039

Linear probe overall acc: 0.3620

Task 6:

Epoch 99, train loss 0.5006, train acc 0.8356, val loss 1.1154, val acc 0.6560

On all previous tasks, loss 9.040341374715169, acc 0.11133333333333334

Norm of the classification rows for the current classes and mean value:

[2.7519972 1.3497381 2.119423  2.7973228 2.7462263 0.5509446 1.5370631
 2.514321  2.8536265 2.1332064]

2.1353867

Norm of the classification rows for the previously seen classes and mean value:
[1.5227425  1.9971836  1.9762008  1.6701982  1.6731459  2.5803008
 1.8089477  1.7417289  1.9385868  2.7700477  1.841221   2.1962938
 2.3451736  1.787723   1.8811346  1.3213485  2.1854184  2.496848
 1.8697784  2.4904058  1.8046441  1.8990395  3.1595752  2.7273653
 2.367117   0.91198856 1.8784467  1.785892   2.9247098  2.6439178
 1.9421747  1.9896591  2.833135   3.871138   2.2231197  1.5917872
 2.0444527  1.8882375  2.561746   4.3624644  3.2811444  2.5053382
 3.0094268  1.1819792  1.0297832  2.3458972  2.368017   3.304519
 2.355299   1.9818263 ]

2.2173655

Sparcity analysis - population sparcity: 0.8965

Linear probe overall acc: 0.3233

Task 7:

Epoch 99, train loss 0.3568, train acc 0.8854, val loss 0.7616, val acc 0.7790

On all previous tasks, loss 9.013311509268624, acc 0.11157142857142857

Norm of the classification rows for the current classes and mean value:

[2.2793949 3.4831214 2.8150861 3.0048308 1.8835701 2.8554049 2.155887
 4.0628314 3.6510625 3.377936 ]
2.9569125

Norm of the classification rows for the previously seen classes and mean value:

[1.9625095 2.445783  2.3140948 1.8686094 2.0102506 3.213728  2.3086061
 2.2470303 2.1899974 3.1708436 2.4580224 2.5554655 3.1436641 2.2043197
 2.4345062 1.8140618 2.6088138 3.1699018 2.1044078 3.594862  2.067562
 2.1716042 3.6308863 3.2042677 2.9022067 2.1305313 2.3334606 2.8103886
 3.4156134 3.18043   2.4829237 2.214312  3.1096563 4.015005  2.4586518
 2.5224924 2.4515417 2.9229128 2.9348218 4.9483075 3.5267415 3.731286
 3.2365787 1.6100866 1.4528815 2.9363613 2.7238708 4.042334  3.5093274
 2.4290414 3.2489843 2.1709309 2.7947972 3.2987769 3.346175  0.5523348
 1.9478292 2.772568  3.1938527 2.6266527]

2.7146409

Sparcity analysis - population sparcity: 0.9061

Linear probe overall acc: 0.2564

Task 8:

Epoch 99, train loss 0.3703, train acc 0.8874, val loss 0.9740, val acc 0.7500

On all previous tasks, loss 6.865471900939942, acc 0.09725

Norm of the classification rows for the current classes and mean value:

[2.8884482 4.096699  3.7922401 2.982943  2.8803632 2.3628986 3.5988529
 3.2348437 2.6851919 3.1387384]

3.166122

Norm of the classification rows for the previously seen classes and mean value:

[2.3082178  2.9670093  2.6579602  2.493237   2.5771704  3.7279785
 3.0129883  2.8668995  2.5617743  3.5674973  3.1489887  2.8011105
 3.8125024  2.7755187  2.8290915  2.186628   2.9761448  3.5712967
 2.4229052  4.012933   2.2816007  2.4366174  3.8808322  4.020108
 3.4607697  2.5493557  2.8646684  3.3627446  3.6954472  4.0343685
 3.010379   2.4866025  3.4003522  4.1901245  2.993267   3.2855244
 2.6823719  3.5266752  3.3336437  5.190445   3.7156405  4.423832
 3.4801207  2.5808358  2.367858   3.270484   3.1293724  4.65455
 4.3013325  3.335085   4.661118   2.590661   3.162181   4.1320806
 3.745722   0.55237293 2.5454311  3.074559   4.0849867  3.371358
 2.651986   4.081176   3.229187   3.933711   2.2721581  3.5270157
 2.7451077  4.477241   4.1926155  3.7594397 ]

3.257328

Sparcity analysis - population sparcity: 0.9197

Linear probe overall acc: 0.2406

Task 9:

Epoch 99, train loss 0.5006, train acc 0.8292, val loss 0.9637, val acc 0.7050

On all previous tasks, loss 10.440846516079374, acc 0.07877777777777778

Norm of the classification rows for the current classes and mean value:

[2.9944518 2.3990598 3.126096  4.471192  3.364417  4.0403647 3.9756348
 2.3182836 4.4620147 5.1582675]

3.630978

Norm of the classification rows for the previously seen classes and mean value:

[2.7387276 3.3655653 2.972989  2.9763606 2.924224  4.1692147 3.400211
 3.1859355 2.8310719 4.1968174 3.7967162 3.0042942 4.4188666 3.1777766
 3.2068045 2.742854  3.4289892 4.313288  3.7012882 4.32255   2.7783573
 2.8524055 4.400719  4.6896443 4.0452924 2.8366916 3.231068  3.7324824
 4.006418  4.552768  3.4546895 2.8793182 3.6462588 4.410965  3.500241
 3.6722896 2.8652842 3.8597152 3.7145402 5.37746   4.1238694 5.0855994
 3.7964206 3.073017  2.7999392 3.6401634 3.4105108 5.2129116 5.0617657
 3.7957575 5.215342  3.0932522 3.7305212 4.416138  4.1534877 0.5575648
 3.318867  3.2984505 4.6037674 3.7575629 3.3086317 4.479947  3.9943864
 4.3265743 2.6500258 3.9362466 3.5251951 4.9485865 5.0323567 4.0519733
 3.1997771 4.8898273 4.150555  3.7476842 3.3194814 2.596795  4.0685987
 3.6371849 3.1442702 3.8853784]

3.7052445

Sparcity analysis - population sparcity: 0.9076

Linear probe overall acc: 0.2239

Task 10:

Training for 100 epochs:

Epoch 99, train loss 0.4850, train acc 0.8346, val loss 1.0466, val acc 0.6980

On all previous tasks, loss 9.817367352294921, acc 0.07

Norm of the classification rows for the current classes and mean value:

[5.071993  5.034712  5.361567  5.4165864 3.2610025 4.2719235 7.5544677
 3.4102814 4.1274204 4.7358623]
4.824581

Sparcity analysis - population sparcity: 0.9183

Linear probe overall acc: 0.2020

| Accuracy    | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 | Task 10 |
|------------|------- |------- |------- |------- |------- |------- |------- |------- |------- |------- |
| Classifier                     | 0.8100 | 0.6100 | 0.4750 | 0.4238 | 0.3620 | 0.3233 | 0.2564 | 0.2406 | 0.2239 | 0.2020 |
| Class                couch    | 0.6000 | 0.3704 | 0.2609 | 0.2727 | 0.2500 | 0.2857 | 0.0000 | 0.1333 | 0.0000 | 0.2083 |
| Class                plate    | 0.9474 | 0.6500 | 0.3810 | 0.2500 | 0.3333 | 0.5000 | 0.2381 | 0.3333 | 0.2381 | 0.2667 |
| Class              leopard    | 0.8636 | 0.5333 | 0.3333 | 0.5200 | 0.3103 | 0.3333 | 0.2308 | 0.3333 | 0.2222 | 0.1667 |
| Class               forest    | 0.7368 | 0.3529 | 0.4667 | 0.5263 | 0.4211 | 0.4706 | 0.3200 | 0.1250 | 0.5000 | 0.3750 |
| Class            streetcar    | 0.8462 | 0.7727 | 0.5926 | 0.4444 | 0.3333 | 0.2308 | 0.2500 | 0.0667 | 0.2609 | 0.2667 |
| Class                 road    | 0.8333 | 0.3333 | 0.7778 | 0.5217 | 0.5882 | 0.4348 | 0.2609 | 0.3158 | 0.3684 | 0.4643 |
| Class                  bee    | 0.7083 | 0.4444 | 0.1667 | 0.2609 | 0.2778 | 0.1053 | 0.2000 | 0.2632 | 0.1364 | 0.2000 |
| Class               beetle    | 0.7778 | 0.4500 | 0.2759 | 0.1739 | 0.1667 | 0.1667 | 0.1579 | 0.2778 | 0.2105 | 0.3000 |
| Class               orchid    | 0.9167 | 0.5200 | 0.3636 | 0.3000 | 0.3462 | 0.3913 | 0.2222 | 0.3182 | 0.2000 | 0.1600 |
| Class             wardrobe    | 0.8462 | 0.8500 | 0.8571 | 0.6875 | 0.5714 | 0.7143 | 0.4211 | 0.5000 | 0.4667 | 0.5000 |
| Class                  ray    |        | 0.6000 | 0.5000 | 0.4545 | 0.3889 | 0.2778 | 0.3636 | 0.2500 | 0.1765 | 0.2353 |
| Class                  sea    |        | 0.6875 | 0.6000 | 0.8421 | 0.4667 | 0.4091 | 0.5000 | 0.3000 | 0.4348 | 0.4286 |
| Class                poppy    |        | 0.7586 | 0.3636 | 0.3600 | 0.1481 | 0.2381 | 0.2400 | 0.1875 | 0.4615 | 0.0833 |
| Class                apple    |        | 0.7391 | 0.5238 | 0.5263 | 0.4167 | 0.3333 | 0.3158 | 0.5600 | 0.4444 | 0.3750 |
| Class                 bear    |        | 0.4615 | 0.4000 | 0.2333 | 0.1429 | 0.0714 | 0.0800 | 0.1667 | 0.0000 | 0.0800 |
| Class           lawn_mower    |        | 0.7778 | 0.6667 | 0.4000 | 0.2857 | 0.5000 | 0.3913 | 0.3000 | 0.5000 | 0.3684 |
| Class             keyboard    |        | 0.6500 | 0.3889 | 0.1429 | 0.2083 | 0.4091 | 0.1290 | 0.0435 | 0.2500 | 0.1875 |
| Class                 tank    |        | 0.9200 | 0.8000 | 0.6250 | 0.3600 | 0.2800 | 0.4000 | 0.1304 | 0.1176 | 0.2500 |
| Class                tiger    |        | 0.5833 | 0.3636 | 0.3200 | 0.2381 | 0.3333 | 0.1600 | 0.3000 | 0.3500 | 0.2667 |
| Class                  can    |        | 0.4667 | 0.4000 | 0.2778 | 0.3636 | 0.3158 | 0.2632 | 0.2667 | 0.1818 | 0.1500 |
| Class            sunflower    |        |        | 0.7059 | 0.3333 | 0.4167 | 0.7333 | 0.2353 | 0.2917 | 0.2000 | 0.3333 |
| Class                clock    |        |        | 0.6667 | 0.2632 | 0.1333 | 0.1111 | 0.2778 | 0.0833 | 0.2308 | 0.0625 |
| Class                tulip    |        |        | 0.4286 | 0.3125 | 0.3571 | 0.2174 | 0.0588 | 0.1429 | 0.2500 | 0.1364 |
| Class               beaver    |        |        | 0.2273 | 0.1500 | 0.0556 | 0.0800 | 0.0000 | 0.1739 | 0.0000 | 0.0800 |
| Class            butterfly    |        |        | 0.3684 | 0.4167 | 0.2105 | 0.1818 | 0.2069 | 0.0800 | 0.0357 | 0.1667 |
| Class                  man    |        |        | 0.5500 | 0.2400 | 0.4000 | 0.2143 | 0.2857 | 0.0870 | 0.1538 | 0.2381 |
| Class              dolphin    |        |        | 0.5000 | 0.5000 | 0.3333 | 0.3333 | 0.2500 | 0.2917 | 0.3636 | 0.0800 |
| Class            palm_tree    |        |        | 0.7143 | 0.7895 | 0.4762 | 0.3810 | 0.3158 | 0.2353 | 0.2727 | 0.1429 |
| Class                  cup    |        |        | 0.6316 | 0.2941 | 0.4375 | 0.4000 | 0.1875 | 0.1053 | 0.3158 | 0.2667 |
| Class               spider    |        |        | 0.5000 | 0.5714 | 0.2727 | 0.2000 | 0.2353 | 0.1905 | 0.1000 | 0.2222 |
| Class           motorcycle    |        |        |        | 0.8636 | 0.6296 | 0.3600 | 0.6190 | 0.6111 | 0.3462 | 0.2609 |
| Class                shrew    |        |        |        | 0.3684 | 0.2500 | 0.3810 | 0.1304 | 0.2353 | 0.1364 | 0.1053 |
| Class                 girl    |        |        |        | 0.4737 | 0.1579 | 0.2273 | 0.1923 | 0.1000 | 0.2222 | 0.1304 |
| Class            cockroach    |        |        |        | 0.6250 | 0.2353 | 0.2500 | 0.2727 | 0.3333 | 0.4211 | 0.2105 |
| Class                train    |        |        |        | 0.4231 | 0.2778 | 0.2500 | 0.1739 | 0.2941 | 0.0870 | 0.1111 |
| Class                table    |        |        |        | 0.6000 | 0.6087 | 0.1481 | 0.1500 | 0.0909 | 0.1429 | 0.0667 |
| Class                  bed    |        |        |        | 0.5833 | 0.2632 | 0.1429 | 0.2273 | 0.0882 | 0.1818 | 0.2353 |
| Class                whale    |        |        |        | 0.3529 | 0.4091 | 0.3750 | 0.2273 | 0.2727 | 0.1905 | 0.1538 |
| Class         sweet_pepper    |        |        |        | 0.2941 | 0.1818 | 0.2143 | 0.0909 | 0.1000 | 0.2105 | 0.1600 |
| Class                plain    |        |        |        | 0.6818 | 0.7333 | 0.6522 | 0.5263 | 0.5000 | 0.5000 | 0.4483 |
| Class             flatfish    |        |        |        |        | 0.3810 | 0.1500 | 0.0588 | 0.2414 | 0.2000 | 0.1739 |
| Class                shark    |        |        |        |        | 0.3077 | 0.2609 | 0.2000 | 0.0435 | 0.1111 | 0.1538 |
| Class           maple_tree    |        |        |        |        | 0.7895 | 0.4783 | 0.3889 | 0.2727 | 0.2857 | 0.3333 |
| Class                 rose    |        |        |        |        | 0.1500 | 0.3793 | 0.1739 | 0.4375 | 0.1111 | 0.0833 |
| Class                 lion    |        |        |        |        | 0.3333 | 0.3793 | 0.2800 | 0.0625 | 0.1000 | 0.2273 |
| Class               orange    |        |        |        |        | 0.6400 | 0.5500 | 0.3333 | 0.5000 | 0.4500 | 0.4737 |
| Class                chair    |        |        |        |        | 0.5263 | 0.3846 | 0.2778 | 0.4167 | 0.3750 | 0.4211 |
| Class              tractor    |        |        |        |        | 0.3333 | 0.3478 | 0.3913 | 0.1579 | 0.1538 | 0.2353 |
| Class               castle    |        |        |        |        | 0.5333 | 0.6667 | 0.4211 | 0.4000 | 0.3684 | 0.1905 |
| Class               possum    |        |        |        |        | 0.4194 | 0.1053 | 0.0909 | 0.0000 | 0.1250 | 0.0526 |
| Class          caterpillar    |        |        |        |        |        | 0.1667 | 0.0909 | 0.1000 | 0.2273 | 0.1176 |
| Class           skyscraper    |        |        |        |        |        | 0.5294 | 0.5417 | 0.4737 | 0.2273 | 0.2941 |
| Class                 bowl    |        |        |        |        |        | 0.4118 | 0.1250 | 0.0667 | 0.1154 | 0.1429 |
| Class                 wolf    |        |        |        |        |        | 0.4000 | 0.2273 | 0.0476 | 0.2353 | 0.1053 |
| Class               rabbit    |        |        |        |        |        | 0.1765 | 0.2500 | 0.1429 | 0.0455 | 0.0000 |
| Class               lizard    |        |        |        |        |        | 0.0769 | 0.1250 | 0.1500 | 0.0000 | 0.0000 |
| Class                 seal    |        |        |        |        |        | 0.1579 | 0.0500 | 0.1176 | 0.0000 | 0.1000 |
| Class                 lamp    |        |        |        |        |        | 0.2500 | 0.1923 | 0.1667 | 0.1500 | 0.1154 |
| Class                 pear    |        |        |        |        |        | 0.4583 | 0.3684 | 0.1250 | 0.1250 | 0.0952 |
| Class                snake    |        |        |        |        |        | 0.2105 | 0.2105 | 0.0476 | 0.1500 | 0.1154 |
| Class                snail    |        |        |        |        |        |        | 0.2632 | 0.0000 | 0.1364 | 0.1154 |
| Class                trout    |        |        |        |        |        |        | 0.4762 | 0.2105 | 0.1429 | 0.2000 |
| Class             oak_tree    |        |        |        |        |        |        | 0.5714 | 0.5000 | 0.4375 | 0.2000 |
| Class         pickup_truck    |        |        |        |        |        |        | 0.3500 | 0.5909 | 0.3333 | 0.0625 |
| Class               turtle    |        |        |        |        |        |        | 0.0625 | 0.0000 | 0.0000 | 0.0435 |
| Class             dinosaur    |        |        |        |        |        |        | 0.4444 | 0.1500 | 0.1250 | 0.1176 |
| Class             kangaroo    |        |        |        |        |        |        | 0.3125 | 0.0556 | 0.1765 | 0.0455 |
| Class                woman    |        |        |        |        |        |        | 0.1429 | 0.1818 | 0.2105 | 0.1000 |
| Class                house    |        |        |        |        |        |        | 0.2857 | 0.1667 | 0.0000 | 0.2500 |
| Class              hamster    |        |        |        |        |        |        | 0.3889 | 0.3000 | 0.1111 | 0.1739 |
| Class                 baby    |        |        |        |        |        |        |        | 0.2800 | 0.1579 | 0.1053 |
| Class        aquarium_fish    |        |        |        |        |        |        |        | 0.4348 | 0.2500 | 0.2963 |
| Class                camel    |        |        |        |        |        |        |        | 0.5000 | 0.1364 | 0.1304 |
| Class                cloud    |        |        |        |        |        |        |        | 0.3571 | 0.2941 | 0.4444 |
| Class              bicycle    |        |        |        |        |        |        |        | 0.5714 | 0.4762 | 0.3158 |
| Class               bottle    |        |        |        |        |        |        |        | 0.3500 | 0.3333 | 0.1429 |
| Class               cattle    |        |        |        |        |        |        |        | 0.4286 | 0.2500 | 0.1053 |
| Class             mushroom    |        |        |        |        |        |        |        | 0.1250 | 0.4444 | 0.0526 |
| Class              raccoon    |        |        |        |        |        |        |        | 0.4762 | 0.1579 | 0.0556 |
| Class                skunk    |        |        |        |        |        |        |        | 0.3636 | 0.2917 | 0.0455 |
| Class           television    |        |        |        |        |        |        |        |        | 0.3478 | 0.3000 |
| Class                 worm    |        |        |        |        |        |        |        |        | 0.3571 | 0.0417 |
| Class            porcupine    |        |        |        |        |        |        |        |        | 0.2353 | 0.2500 |
| Class            telephone    |        |        |        |        |        |        |        |        | 0.4211 | 0.3600 |
| Class            pine_tree    |        |        |        |        |        |        |        |        | 0.4167 | 0.2857 |
| Class             elephant    |        |        |        |        |        |        |        |        | 0.3158 | 0.0909 |
| Class                mouse    |        |        |        |        |        |        |        |        | 0.0500 | 0.2143 |
| Class                otter    |        |        |        |        |        |        |        |        | 0.0417 | 0.0769 |
| Class                  boy    |        |        |        |        |        |        |        |        | 0.2222 | 0.1481 |
| Class          willow_tree    |        |        |        |        |        |        |        |        | 0.2105 | 0.2917 |
| Class                  fox    |        |        |        |        |        |        |        |        |        | 0.2143 |
| Class                  bus    |        |        |        |        |        |        |        |        |        | 0.3000 |
| Class             mountain    |        |        |        |        |        |        |        |        |        | 0.4091 |
| Class             squirrel    |        |        |        |        |        |        |        |        |        | 0.0833 |
| Class              lobster    |        |        |        |        |        |        |        |        |        | 0.0526 |
| Class                 crab    |        |        |        |        |        |        |        |        |        | 0.1667 |
| Class               bridge    |        |        |        |        |        |        |        |        |        | 0.2174 |
| Class            crocodile    |        |        |        |        |        |        |        |        |        | 0.2667 |
| Class               rocket    |        |        |        |        |        |        |        |        |        | 0.4118 |
| Class           chimpanzee    |        |        |        |        |        |        |        |        |        | 0.5455 |


#### Forgetting on the latent representation

Comparing MNIST and CIFAR, on the forgetting (on the latent representation) of the items learned on the first task, over the next 4 tast:

* the values are normalized by the average classification accuracy of the items when first learned.

MNIST:

[1.0, 0.9524859663191659, 0.9464715316760224, 0.9246692060946271, 0.9354951884522854]

CIFAR:

[1.0, 0.8386904761904761, 0.7927380952380952, 0.68775, 0.6867738095238094]



