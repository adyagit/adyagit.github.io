
# Python Sklearn Decision tree Classifier


```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../Proc_FAQ/modeoftransport.csv")
print(df.head(10))

```

       Gender  Car Owenership Travel Cost $/km Income Level Transportation mode
    0    male               0            cheap          low                 bus
    1    male               1            cheap       medium                 bus
    2  female               1            cheap       medium               train
    3  female               0            cheap          low                 bus
    4    male               1            cheap       medium                 bus
    5    male               0         standard       medium               train
    6  female               1         standard       medium               train
    7  female               1        expensive         High                 car
    8    male               2        expensive       medium                 car
    9  female               2        expensive         High                 car



```python
col = df.columns
enc = LabelEncoder()
df = df.apply(enc.fit_transform)

target = df['Transportation mode']
df.drop(['Transportation mode'],axis=1, inplace=True)
#Encoded features
print(df.head(10))
#Encoded target
print(target)
```

       Gender  Car Owenership  Travel Cost $/km  Income Level
    0       1               0                 0             1
    1       1               1                 0             2
    2       0               1                 0             2
    3       0               0                 0             1
    4       1               1                 0             2
    5       1               0                 2             2
    6       0               1                 2             2
    7       0               1                 1             0
    8       1               2                 1             2
    9       0               2                 1             0
    0    0
    1    0
    2    2
    3    0
    4    0
    5    2
    6    2
    7    1
    8    1
    9    1
    Name: Transportation mode, dtype: int64



```python
dtree=DecisionTreeClassifier()
dtree.fit(df,target)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
import numpy as np
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(df.shape[1]):
    print("%d. feature %d (%s) (%f) " % (f + 1, indices[f],df.columns[indices[f]], importances[indices[f]]))
```

    Feature ranking:
    1. feature 2 (Travel Cost $/km) (0.757576) 
    2. feature 3 (Income Level) (0.151515) 
    3. feature 0 (Gender) (0.090909) 
    4. feature 1 (Car Owenership) (0.000000) 



```python
from sklearn.tree import export_graphviz
import io
export_graphviz(dtree, out_file='../Downloads/dtree.out',
                feature_names=col,
                filled=True, rounded=True,
                special_characters=True) 

```

<img src="../Proc_FAQ/tree.png" width=50%>


```python

```
