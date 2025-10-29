Notes with latest updates, for

*October, 29*

### Comparing cross-entropy training and (in infancy) domain adversarial training

#### Latent space, task 1 [1, 2]:

##### Cross-entropy

![pattern](./images_mnist/ce_rehearsal_layer2_ground_truth.png)

##### Domain adaptation

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_task1.png)

#### Latent space, task 2 [3, 4]:

#### Cross-entropy

* Ground truth:

![pattern](./images_mnist/ce_rehearsal_layer2_ground_truth_task2.png)

* Predictions:

![pattern](./images_mnist/ce_rehearsal_layer2_pred_task2.png)

##### Domain adaptation

* Ground truth

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_task2.png)

* Predictions:

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_preds_layer3_task2.png)



#### Latent space, task 2 [9, 0]:

#### Cross-entropy

* Ground truth:

![pattern](./images_mnist/ce_rehearsal_layer2_ground_truth_task5.png)

* Predictions:

![pattern](./images_mnist/ce_rehearsal_layer2_pred_task5.png)

##### Domain adaptation

* Ground truth

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_task5_balanced.png)

* Predictions:

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_preds_task5_balanced.png)


* Predictions domain, Latent space, layer 3, digits and patterns

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_preds_domain_task5_balanced.png)

* Forgetting

![pattern](./images_mnist/adversarial/single_input_rehearsal/forgetting_task5_balanced.png)

#### Comparing with cross-entropy / domain adaptation / contrastive learning

![pattern](./images_mnist/ce_rehearsal_layer2_ground_truth_task5.png)

![pattern](./images_mnist/adversarial/single_input_rehearsal/latent_layer3_task5_balanced.png)

![pattern](./images_mnist/sequential_contrastive_rehearsal_task5_latent.png)



#### Two things seem to be necessary

1. Keep the logits alive for classes trained previously
2. Separate in the latent space classes not seen together
