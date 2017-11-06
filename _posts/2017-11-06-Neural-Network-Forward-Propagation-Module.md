---
layout: post
category: python machine_learning deep_learning 
---

The best part about Andrew Ng's Deep Learning course was how elegantly he broke down the whole network into its constituents. In the previous post I've posted his explanation of how to initialize the parameters. That is the first step of building a model. In this post we will see how the Forward Propagation module works. This module will be broken down into three steps 

- LINEAR
- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
- [LINEAR -> RELU] $$\times$$ (L-1) -> LINEAR -> SIGMOID (whole model)

### Linear Forward
The LINEAR function is the one that generates the logits. This function will be called linear_forward. This will take the activation from the previous layer, the weights of the current layer and the biases of the current layer and generate the input for the activation function also called the ***pre-activation*** parameter or the ***logits***
The function returns this preactivation parameter and a cache of the inputs. This cache will be used during the back prop step.



```python
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): 
         (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape 
         (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape 
         (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, 
         also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; 
         stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
```

### Linear Forward Activation

The output from the previous function is fed into the activation function int this step. The activation function used for the **L-1** hidden layers will be **RELU** . The last layer will use a **SIGMOID** activation since the output from that will be between 0 and 1. This makes it easier to define a cost function that can be used for our weights  optimization.

- **ReLU**: The mathematical formula for ReLu is $$A = RELU(Z) = max(0, Z)$$. 
- **Sigmoid**: $$\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$$

These functions return **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`"

The function implementation will have a LINEAR forward step followed by an ACTIVATION forward step.

Mathematical relation is: 
$$A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$$ where the activation "g" can be sigmoid() or relu()


```python
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): 
              (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape 
         (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape 
         (size of the current layer, 1)
    activation -- the activation to be used in this layer, 
                  stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, 
         also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
        
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
```

### Linear Forward Activation for ***L*** Layers

The above function works well for a single layer. For a L-Layer model we will have to loop through each layer and do the forward prop. 
**L-1** layers will have the **RELU** activation and the last layer will have the **SIGMOID** activation. 
We need to append the cache from each layer to a bigger array "`caches`" for use during the back prop step. 


```python

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        
        A, cache = linear_activation_forward(A_prev, parameters["W"+ str(l)], parameters['b'+ str(l)], activation='relu')
        caches.append(cache)
        
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    
    AL, cache = linear_activation_forward(A,parameters["W"+ str(L)], parameters['b'+ str(L)], activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
```
