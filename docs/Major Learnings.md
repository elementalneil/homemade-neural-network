## Implementation Post Chapter 2

### Notes

1. ReLU often learns faster than sigmoid in this network, so starting with a lower learning rate for ReLU can make sense. A learning rate around 1/10th of the sigmoid learning rate works well for me.

- Commit: `7176c4cbb9f6e7599c5bb7452fc2a5434ca0c8fe`
- Subject: Updated runner program to use package

---

## Learnings during Chapter 3 Implementation

### Notes

1. The final layer should produce raw logits (no sigmoid or ReLU), then apply softmax to convert those logits into class probabilities. Applying sigmoid first squashes every logit into (0, 1); for 10 classes, softmax of those squashed values cannot assign more than about 23% probability to one class.

2. With sigmoid activation and binary cross-entropy, the output-layer derivative simplifies to `A - Y`.

   For one example and one output unit, the binary cross-entropy cost is:

   $$
   c(a, y) = -[y\ln(a) + (1-y)\ln(1-a)]
   $$

   Its derivative with respect to the activation $a$ is:

   $$
   \frac{\partial c}{\partial a}
   = -\frac{y}{a} + \frac{1-y}{1-a}
   $$

   The sigmoid derivative is:

   $$
   \frac{\partial a}{\partial z} = a(1-a)
   $$

   Applying the chain rule and simplifying:

   $$
   \begin{aligned}
   \frac{\partial c}{\partial z}
   &= \frac{\partial c}{\partial a}\frac{\partial a}{\partial z}\\
   &= \left(-\frac{y}{a}+\frac{1-y}{1-a}\right)a(1-a)\\
   &= -y(1-a)+(1-y)a\\
   &= a-y
   \end{aligned}
   $$

   Therefore, the output-layer error can be computed directly as `A - Y`, without multiplying by the sigmoid derivative again. For a batch-averaged cost, the gradients are averaged over the number of examples; this implementation applies that factor as `1 / n` when calculating weight and bias gradients.

   This simplification applies when sigmoid activation is paired with binary cross-entropy. It does not apply unchanged to other activation and loss combinations.