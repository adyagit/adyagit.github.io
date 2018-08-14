



# Understanding the math behind Decision Tree Classifier

Consider a dataset as shown below. The example is from [revoledu](http://people.revoledu.com/kardi/tutorial/DecisionTree/how-to-measure-impurity.htm)

<img src="../Proc_FAQ/modeoftransport.png" width=50%>

The task is to be able to classify whether a set of measured attributes will fall into one of the three classes Bus, Train or Car as their mode of transport.  One of the measures used for this task is a measure of homogenity of the dataset or a subset of the dataset. ***Homogenous*** or ***pure** data is if all the datapoints belong to the same class. Else its ***heterogenous*** or ***impure***. 

There are several indices to measure impurity. We will look at **Gini Index**. The other popular ones are Entropy and Classification error. 

If $p_j$ is the probability of class j then Gini Index is defined as 

$$ Gini Index = 1 - \sum_{j=1}^{k} p_j^2 $$

So for the dataset as a whole above, the Gini Index for the distribution 4 busses, 3 cars and 3 trains will be 

$$ Prob (bus) = \frac{4}{10} $$
$$ Prob (car) = \frac{3}{10} $$
$$ Prob (train) = \frac{3}{10} $$

$$ Gini Index = 1 - (Prob(bus)^2 + Prob(car)^2 + Prob(train)^2)$$
$$ \implies 1 - (0.16+0.09+0.09) = 0.66 $$

In general the decision tree algorithm is recursive. It tries to find the best split of the data such that the purity of the leaf node is maximized. For example lets take subsets of the above dataset and compute the gini index 
<table align='center'>
    <tr>
        <td>
           <img src="../Proc_FAQ/cost.png" width=70%>
        </td>
        <td>
           <img src="../Proc_FAQ/own.png" width=70% >
         </td>
     </tr>
     <tr>
         <td>
           <img src="../Proc_FAQ/gender.png" width=70% >
        </td>
        <td>
           <img src="../Proc_FAQ/income.png" width=70%>
         </td>
    </tr>
 </table>



We need to further seperate out the classes from each attribute and compute the impurity index
<img src="../Proc_FAQ/cost_class.png" width=20% >

Gini Index for class Cheap will be 

$$ prob(bus) = 4/5 $$
$$ prob(train) = 1/5 $$

$$ Gini Index = 1 -((4/5)^2 + (1/5)^2)  = 0.32 $$

Gini Index for class Standard will be 

$$ prob(train) = 2/2 $$

$$ Gini Index = 1 -((1)^2 )  = 0 $$

Gini Index for class Expensive will be 

$$ prob(car) = 3/3 $$

$$ Gini Index = 1 -((1)^2 )  = 0 $$

We are interested in finding out the difference in impurity before splitting the table and after splitting the table based on the values of an attribute i. This difference is our information gain. 

$$ Information Gain = Gini Index(Parent Data) - \sum \frac{k}{n}Gini Index(Attribute_i) $$

In the above case 

$ Information Gain(Total Cost (\$/km)) = 0.66 - (5/10*0.32 + 3/10*0 + 2/10*0)
\implies 0.5 $

Similarly continuing for other attributes 
<img src="../Proc_FAQ/gini.png" width=70% >

Consolidating all the Information Gains 
<table>
<tr>
        <td>
            Attribute
        </td>

         <td>
         Information Gain
        </td>
     </tr>
    <tr>
        <td>
            Travel Cost (dollar/km)
        </td>

         <td>
         0.5
        </td>
     </tr>
     <tr>
        <td>
            Gender
        </td>

         <td>
         0.06
        </td>
       </tr>
       <tr>
        <td>
            Car Owenership
        </td>

         <td>
         0.207
        </td>
        </tr>
        <tr>
        <td>
            Income Level
        </td>

         <td>
         0.293
        </td>
        
    </tr>
</table>

From the above we pick the first split that maximizes our information gain i.e our first split will be Travel Cost

Looking at the filtered data we already have two leaves that are pure. The next split needs to be for the branch that has the class ***cheap*** 

<img src="../Proc_FAQ/first_split.png" width=40% >


## Random Forest

<img src="../Proc_FAQ/random-forest.png" width=70% >
[image source](https://nkimberly.wordpress.com/2018/03/17/implementing-regressions-in-python-support-vector-machine-decision-tree-cart-and-random-forest/)



```python

```
