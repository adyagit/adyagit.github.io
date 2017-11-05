---
layout: post
category: python machine_learning shallow_net
title: Multinomial Logistic Classifier
---

Logistic classifier is a  Linear Classifier. It takes a linear function ***X*** and multiples it with a Weight matrix ***W***. The outputs are referred to as scores.


$$WX + b = y $$


Here ***b*** is the bias vector. The score or Logit **[y]** vector will have the highest value for the class that the sample belongs to and low for every other class. For a Logistic Classifier we need to turn the scores to ***probabilities***. This is done using the ***softmax*** function. The *softmax* function is as shown below

$$\begin{equation}
S(y_i) = \frac{e^{y_i}}{\sum\limits_{j} e^{y_j}}
\end{equation}$$

This function takes high scores and converts it to a probability very close to 1 and assigns a low probability ~ 0 for low scores. The probabilities sum to 1.


```python
"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    prob = np.exp(x) / np.sum(np.exp(x),axis=0)
    return prob

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
```


![png](/snippets/public/MultinomialLogisticClassifier_1_0.png)


As seen from this plot, the three vectors are defined such that the first class has the highest score at the end of the range.
For large negative values the class probability is very close to  zero. As the score gets positive and larger the probability
gets closer to 1.


The true labels are then encoded using ***One-Hot Encoding***.

One-hot Encoding assigns 1 for the correct class and exactly zero for all other classes for the given sample

Now to assess our prediction we can compare our probability predictions to this one hot encoded vector using the distance measure ***Cross-Entropy***

$$\begin{equation}\begin{bmatrix}
p_{1} \\
p_{2} \\
p_{3}
\end{bmatrix} \overset{Cross-Entropy}{D(S,L) = \sum\limits_{i}{L_{i}log(S_i)}}\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
\end{equation}$$

The overall flow of the algorithm would then be

$$\begin{equation}
\overset{X - Input}{\begin{bmatrix}
x_{1}\\
x_{2}\\
x_{3}
\end{bmatrix}}\hspace{1cm} \overset{Linear Model}{WX + b} \hspace{1cm} \overset{Logits}{\begin{bmatrix}
s_{1} \\
s_{2} \\
s_{3}
\end{bmatrix}}\hspace{1cm} \overset{Softmax}{S(y)}\hspace{1cm} \overset{Proper-Probabilities}{\begin{bmatrix}
p_{1} \\
p_{2} \\
p_{3}
\end{bmatrix}} \hspace{1cm}\overset{Cross-Entropy}{D(S,L)} \hspace{1cm}\overset{1-Hot Labels}{\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}}
\end{equation}
$$
Now the objective is to find the weights ***W*** and biases ***b***. The best weights and biases are those that give low distance for correct class prediction and high distance for a wrong prediction. This now becomes a funtion minimization task.

The Loss function ***£*** we are trying to minimize is the global average cross entropy given by

$$\begin{equation}
£ = \frac{1}{N} \sum\limits_{i}D(S(wx_i + b),L_i)
\end{equation}$$


It is always advisable to have a zero mean and equal variance for all the input variables when ever possible to avoid numerical
instability and also to condition the model so that the minima can be easily found by the optimizer.

Also it is important to start with a good initial guess for the weights and the biases. This is ensured by picking the initial guess from a gaussian distribution with mean 0 and standard deviation **$$\sigma$$**. Usually a small value of **$$\sigma$$** is preferred as opposed to a large **$$\sigma$$** since these can be more opinionated. I dont fully understand why but lets accept it for now.

The optimization algorithm now computes the gradients of the loss function with respect to the weights and biases. The weights and biases are moved in a direction opposite to these gradients. The step size $$\alpha$$ is called the learning rate.

$$\begin{equation}
w \Leftarrow w - \alpha \Delta_{w} £\\
b \Leftarrow b - \alpha \Delta_{b} £\\
\end{equation}$$

Finally for ***N*** inputs and ***K*** outputs we have

$$\begin{equation}
(N+1)K
\end{equation}$$ parameters to solve for


```python

```
