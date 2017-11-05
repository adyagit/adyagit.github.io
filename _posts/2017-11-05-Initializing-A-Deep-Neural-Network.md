---
layout: post
title: Initializing A Deep Neural network
category: python machine_learning deep_nets Initializing
---

This snippet is from Andrew Ng 's coursera course on deep learning.

One of the first tasks we need to do in building a deep network is to initialize the parameters properly. The helper function below takes the dimensions of the network in terms of number of hidden units in each layer and initializes the parameters accordingly. The initialized parameters are stored in a dictionary to make for easy retrieval.

Weights are randomly initialized and biases are set to zero.

The table below shows an example network dimensions.

For a L layer Neural, where $$n^{[l]}$$ is the number of units in layer $$l$$, if the size of our input $$X$$ is $$(12288, 209)$$ (with $$m=209$$ examples) then

|           | **Shape of W** | **Shape of B** | **Activation** | **Shape of Activation**|
|:---------:|----------------|----------------|----------------|------------------------|
|**Layer 1** |$$(n^{[1]},12288)$$|$$(n^{[1]},1)$$|$$Z^{[1]} = W^{[1]}  X + b^{[1]} $$|$$(n^{[1]},209)$$|
|**Layer 2** |$$(n^{[2]}, n^{[1]})$$|$$(n^{[2]},1)$$|$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$|$$(n^{[2]}, 209)$$|
|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|$$\vdots$$|
|**Layer L-1**|$$(n^{[L-1]}, n^{[L-2]})$$|$$(n^{[L-1]}, 1)$$|$$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$$|$$(n^{[L-1]}, 209)$$|
|**Layer L**|$$(n^{[L]}, n^{[L-1]})$$|$$(n^{[L]}, 1)$$|$$Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$$|$$(n^{[L]}, 209)$$|
       

```python
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
 

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))


    return parameters
```
