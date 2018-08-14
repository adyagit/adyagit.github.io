
# Linear Regression 

This function fits a polynomial LinearRegression model on the training data X_train for degrees 1, 3, 6, and 9. 
<br>

PolynomialFeatures in sklearn.preprocessing lets us create the polynomial features. For each model, we will predict 100 values over the interval x = 0 to 10 (np.linspace(0,10,100)) and store this in a numpy array. The first row of this array corresponds to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
<br>

The block below generates a dataset of a sinusoidal wave and splits it into train and test set


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

X_train = X_train.reshape(11,1)
y_train = y_train.reshape(11,1)
X_test = X_test.reshape(4,1)
y_test = y_test.reshape(4,1)
```

In the block below we are fitting polynomials of order 1,3,6 and 9 

Polynomial regression is a special case of linear regression . In Linear regression we are trying to solve for one coefficient 

$$ y = a X $$

In polynomial regression we are solving for 

$$ y = a_1X + a_2 X^2 + a_3 X^3 ... $$

sklearn's PolynomialFeatures takes a first order input and returns a matrix with each element representing the higher order term. For example


```python
from sklearn.preprocessing import PolynomialFeatures

A = [1,2]
print("input array A:")
print(A)

print(PolynomialFeatures(degree=3,include_bias=False).fit_transform(A))


```

    input array A:
    [1, 2]
    [[ 1.  2.  1.  2.  4.  1.  2.  4.  8.]]


    /nobackup/applications/anaconda2_python2.7/lib/python2.7/site-packages/sklearn/utils/validation.py:395: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)
    /nobackup/applications/anaconda2_python2.7/lib/python2.7/site-packages/sklearn/utils/validation.py:395: DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
      DeprecationWarning)


We can now fit several Polynomial Regression models and get predictions out of it . sklearn provides a LinearRegression function for this purpose. A plot function below shows how the predicted values compare as against the input 


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import r2_score

arr = []
r2_pred = []
for count, deg in enumerate([1,3,6,9]):
    poly = PolynomialFeatures(degree=deg,include_bias=False)
    X = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X, y_train)
    predict = poly.fit_transform(np.linspace(0, 10, 100).reshape(100,1))        
    y_plot = model.predict(predict)
    y_pred = model.predict(X)
    r2_pred.append(r2_score(y_train,y_pred))
    arr.append(y_plot)
arr = np.asarray(arr).reshape(4,100)
print(r2_pred)

```

    [0.42924577812346643, 0.58719953687798487, 0.9901823324795076, 0.99803706255439639]



```python
def plot_compare(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.figure(figsize=(10,5))
    plt.bar([1,3,6,9],r2_pred)
    plt.xticks([1,3,6,9],['deg=1','deg=3','deg=6','deg=9'])
```


```python
%matplotlib inline
plot_compare(arr)
```


![png](2_linearRegression_files/2_linearRegression_7_0.png)



![png](2_linearRegression_files/2_linearRegression_7_1.png)


## Loss Function for Regression

General $ R^2 $ loss function for linear regression is written as 
$$ RSS = \sum_{i=1}^{N} (y_i - (W*x+b))^2 $$

In order to avoid overfitting, we can penalize terms with very large weights. This is done by a few different ways 

### Ridge Regression 

$$ RSS = \sum_{i=1}^{N} (y_i - (W*x+b))^2 + \alpha \sum_{j=1}^{k} w_j^2$$

### Lasso Regression

$$ RSS = \sum_{i=1}^{N} (y_i - (W*x+b))^2 + \alpha \sum_{j=1}^{k}|w_j| $$




```python

```
