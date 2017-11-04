
# Nonlinear Decision Boundaries
Multinomial Logistic Classifier is a linear classifier. For linear classifiers the decision boundaries are linear like the one shown below. It partitions the feature space using a linear function. This might not always work for us.


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

s = np.random.uniform(-1,0,1000)
t = np.arange(len(s))
t2 = t+1500
sns.regplot(t,s,fit_reg=False)
sns.regplot(t2,s,fit_reg=False)
plt.axvline(x=1250,color='k')
plt.show()

```


![png](NonlinearClassifiers_files/NonlinearClassifiers_1_0.png)


For the distribution below there is no linear decision boundary that seperates out the blue points from the green. For such cases Linear Classifiers will not work.


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N))
 
data = (g1, g2, g3)
colors = ("blue", "green", "blue")
groups = ("coffee", "tea", "water") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
 
for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)


plt.show()
```


![png](NonlinearClassifiers_files/NonlinearClassifiers_3_0.png)


We can create additional features that allow for more complex decision boundaries. Like in the figure shown below, the author combines three step functions to get a cubic decison boundary. 

<img src="neural_net_act_func.png" alt="Neural Net" style="width: 800px;"/>

These actuation functions are chosen in such a way that they have a smooth derivative like shown below.
<img src="neural_net_act_func_smooth.png" alt="Neural Net" style="width: 800px;"/>

The common choises are 

## Step Function


```python
x = np.arange(-6,6)
y = np.where(x>0,1,0)

sns.set_style('white')
plt.step(x,y)
plt.ylim((-0.2,1.2))
plt.scatter([0,0],[0,1],facecolors='none')
plt.grid('on')
plt.show()

```


![png](NonlinearClassifiers_files/NonlinearClassifiers_6_0.png)


This step function activates the neurons (sets output to 1) if Y > 0 else the neuron is not activated. This works best for a binary classifier. But for a multiclass classifer all Neurons would end up resulting in 1. Hence they are best only for a binary classifier.

## Linear function

For a multiclass classifier we would need the ability to say how much a neuron gets activated because of each input. This can be achieved by a simple linear activation function 

\begin{equation}
Y = cX
\end{equation}



```python
x = np.arange(-6,6)
y = 100*x

sns.set_style('white')
plt.plot(x,y)
plt.annotate('Y = c*X', xy=(1, 1), xytext=(1, 1.5))
plt.grid('on')
plt.show()
```


![png](NonlinearClassifiers_files/NonlinearClassifiers_8_0.png)



For a linear function the activation is proportional to input . This way we get a a range of activations. If more than one fires we can pick the max of them and pick that. But the problem is with gradient descent. The derivative is a constant. So the changes made by the backp propagation is independent of the input. 

Also, there is no real need for multiple layers. We can combine all hidden layers and represent it as one layerwith one linear activation function. 

## Sigmoid Function


```python
import math
x = np.arange(-6,6,0.1)
y = 1/(1+np.exp(-x))

sns.set_style('white')
plt.plot(x,y)

plt.axvline(x=0, color='k',linestyle='dashdot')
plt.axhline(y=0.5, color='k',linestyle='dashdot')
#plt.annotate('Y = c*X', xy=(1,1), xytext=(1, 1.5))
plt.grid('on')
plt.show()
```


![png](NonlinearClassifiers_files/NonlinearClassifiers_10_0.png)


Its a smooth and “step like” function. It is nonlinear in nature. Combinations of this function are also nonlinear. Now we can stack layers.It will give an analog activation unlike step function. It has a smooth gradient too.

Another advantage of this activation function is, unlike linear function, the output of the activation function is always going to be in range (0,1) compared to (-inf, inf) of linear function. So we have our activations bound in a range. Nice, it won’t blow up the activations then.

Sigmoid functions are one of the most widely used activation functions today. Towards either end of the sigmoid function though, the Y values tend to respond very less to changes in X.The gradient at that region is going to be small. It gives rise to a problem of “vanishing gradients”. 

Gradient is small or has vanished ( cannot make significant change because of the extremely small value ). The network refuses to learn further or is drastically slow ( depending on use case and until gradient /computation gets hit by floating point value limits ). There are ways to work around this problem and sigmoid is still very popular in classification problems.

## Tanh Function


```python
import math
x = np.arange(-6,6,0.1)
y = 2/(1+np.exp(-2*x)) -1

sns.set_style('white')
plt.plot(x,y)

plt.axvline(x=0, color='k',linestyle='dashdot')
plt.axhline(y=0, color='k',linestyle='dashdot')
#plt.annotate('Y = c*X', xy=(1,1), xytext=(1, 1.5))
plt.grid('on')
plt.show()
```


![png](NonlinearClassifiers_files/NonlinearClassifiers_12_0.png)


Tanh is very similar to sigmoid. The function is smooth and non linear. The bounds are between (-1 and 1) though. The gradients are steeper than sigmoid

## ReLU

ReLu (rectified linear unit) is nonlinear in nature. And combinations of ReLu are also non linear! ( in fact it is a good approximator. Any function can be approximated with combinations of ReLu). It is not bound though. The range of ReLu is (0, inf). This means it can blow up the activation.




```python
import math
x = np.arange(-6,6,0.1)
y = np.where(x>0,x,0)

sns.set_style('white')
plt.plot(x,y)

plt.axvline(x=0, color='k',linestyle='dashdot')
plt.axhline(y=0, color='k',linestyle='dashdot')
#plt.annotate('Y = c*X', xy=(1,1), xytext=(1, 1.5))
plt.grid('on')
plt.show()
```


![png](NonlinearClassifiers_files/NonlinearClassifiers_15_0.png)


Imagine a big neural network with a lot of neurons. Using a sigmoid or tanh will cause almost all neurons to fire in an analog way. That means almost all activations will be processed to describe the output of a network. In other words the activation is dense. This is costly. We would ideally want a few neurons in the network to not activate and thereby making the activations sparse and efficient.

ReLu gives us this benefit. Imagine a network with random initialized weights ( or normalised ) and almost 50% of the network yields 0 activation because of the characteristic of ReLu ( output 0 for negative values of x ). This means a fewer neurons are firing ( sparse activation ) and the network is lighter. 

Because of the horizontal line in ReLu( for negative X ), the gradient can go towards 0. For activations in that region of ReLu, gradient will be 0 because of which the weights will not get adjusted during descent. That means, those neurons which go into that state will stop responding to variations in error/ input ( simply because gradient is 0, nothing changes ). This is called dying ReLu problem. This problem can cause several neurons to just die and not respond making a substantial part of the network passive. 

There are variations in ReLu to mitigate this issue by simply making the horizontal line into non-horizontal component . for example y = 0.01x for x<0 will make it a slightly inclined line rather than horizontal line. This is leaky ReLu. There are other variations too. The main idea is to let the gradient be non zero and recover during training eventually.

ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations. That is a good point to consider when we are designing deep neural nets.
