---
layout: post
---

## This class will have functions used for curve manipulations

Will post the link to the code here soon.


# Smooth curve and test derivative

Unit Test



```python
cf = curve_functions()

df1 = cf.smooth()
df = cf.get_curve()


fig, ax = plt.subplots()
plt.hold(True)
df1['Force2'].plot(ax=ax,label='Smooth')
plt.legend()

ax.autoscale(enable=True)
df['Force2'].plot(ax=ax,label='Raw')
plt.legend()
plt.show()

cf2 = curve_functions(data=df1)
df2 = cf2.derivative()
fig, ax = plt.subplots()
df2['Force2'].plot(ax=ax,label='Smooth Derivative')
plt.hold(True)

plt.legend()
cf2 = curve_functions(data=df)
df2 = cf2.derivative()



ax.autoscale(enable=True)
df2['Force2'].plot(ax=ax,label='Raw Derivative')
plt.legend()
plt.show()
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_4_1.png)


                     Force    Force2
    displacement                    
     0.000000    -2.802715  7.197285
    -0.080025    -3.227336  6.772664
    -0.161868    -3.670054  6.329946
    -0.248095    -4.139189  5.860811
    -0.339103    -4.635345  5.364655
                     Force     Force2
    displacement                     
     0.000000     0.000000  10.000000
    -0.080025    -0.779175   9.220825
    -0.161868    -2.938103   7.061897
    -0.248095    -2.820479   7.179521
    -0.339103    -3.028321   6.971679
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_4_3.png)


# Extrapolate using last slope
unit test



```python
cf = curve_functions()
x = np.arange(-8,-11,-1)

df1 = cf.extrapolate(x)
print(df1.tail())
df = cf.get_curve()
fig, ax = plt.subplots()
plt.hold(True)
df1.plot(ax=ax,label='extrapolated',linewidth=0.5,style='-.')
#plt.legend()


df.plot(ax=ax,label='Raw')
ax.autoscale(enable=True)
plt.legend()
plt.show()
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
                       Force      Force2
    displacement                        
    -6.986976     -89.248275  -79.248275
    -7.006574     -91.898155  -81.898155
    -8.000000    -226.221029 -216.221029
    -9.000000    -361.432786 -351.432786
    -10.000000   -496.644542 -486.644542
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_6_1.png)


# Digitize Curve

unit test


```python
cf = curve_functions()
x = np.arange(-8,-11,-1)

df1 = cf.digitize(10)
print(df1.tail())
df = cf.get_curve()
fig, ax = plt.subplots()
plt.hold(True)
df1.plot(ax=ax,label='digitized',linewidth=0.5,marker='o')
plt.legend()


df.plot(ax=ax,label='Raw')
ax.autoscale(enable=True)
plt.legend()
plt.show()
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
                      Force     Force2
    displacement                      
    -3.892541    -31.101336 -21.101336
    -4.671049    -37.491759 -27.491759
    -5.449558    -45.334357 -35.334357
    -6.228066    -54.173611 -44.173611
    -7.006574    -91.898155 -81.898155
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_8_1.png)



```python
cf = curve_functions()
df1 = cf.integrate()
print(df1.tail())
df = cf.get_curve()
fig, ax = plt.subplots()
plt.hold(True)
df1.plot(ax=ax,label='integral',linewidth=0.5)
plt.legend()


#df.plot(ax=ax,label='Raw')
#ax.autoscale(enable=True)
#plt.legend()
plt.show()
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
                       Force      Force2
    displacement                        
    -6.922191     196.973204  127.751294
    -6.945518     198.901371  129.446191
    -6.967017     200.733645  131.063475
    -6.986976     202.487777  132.618017
    -7.006574     204.262830  134.197090
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_9_1.png)



```python
cf = curve_functions()
help(cf.resultant)
help(cf.add)
df1 = cf.resultant(col=['Force','Force2'])
print(df1.tail())
df = cf.get_curve()
fig, ax = plt.subplots()
plt.hold(True)
df1.plot(ax=ax,label='multiply',linewidth=0.5)
plt.legend()


df.plot(ax=ax,label='Raw')
ax.autoscale(enable=True)
plt.legend()
plt.show()
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
    Help on method resultant in module __main__:
    
    resultant(col=None) method of __main__.curve_functions instance
        Takes column input as col=['col1','col2'...] and returns the resultant as
        
                    res = (df[col1]**2+df[col2]**2+...)**0.5
        
        If col=None returns resultant of all columns in the dataset
    
    Help on method add in module __main__:
    
    add(col=None) method of __main__.curve_functions instance
        Takes column input as col=['col1','col2'...] and returns the result as
        
                    res = (df[col1]+df[col2]+...)
        
        If col=None adds all columns in the dataset
    
    displacement
    -6.922191    108.262433
    -6.945518    111.843030
    -6.967017    115.510712
    -6.986976    119.354697
    -7.006574    123.095811
    dtype: float64
    


![png](CurveManipulationFunctions_files/CurveManipulationFunctions_10_1.png)



```python
cf = curve_functions()
help(cf.resultant)
help(cf.add)

df1 = cf.curve_lookup(col=['Force'],x=-6,y=-50)
```

    Select the column name to be set as index: 
    Index(['displacement', 'Force', 'Force2'], dtype='object')
    displacement
    Visulize plot[y/n]? n
    Help on method resultant in module __main__:
    
    resultant(col=None) method of __main__.curve_functions instance
        Takes column input as col=['col1','col2'...] and returns the resultant as
        
                    res = (df[col1]**2+df[col2]**2+...)**0.5
        
        If col=None returns resultant of all columns in the dataset
    
    Help on method add in module __main__:
    
    add(col=None) method of __main__.curve_functions instance
        Takes column input as col=['col1','col2'...] and returns the result as
        
                    res = (df[col1]+df[col2]+...)
        
        If col=None adds all columns in the dataset
    
    Force    0.0
    Name: 0.0, dtype: float64 Force   -0.779175
    Name: -0.080025, dtype: float64
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-54-8b3313d9a903> in <module>()
          3 help(cf.add)
          4 
    ----> 5 df1 = cf.curve_lookup(col=['Force'],x=-6,y=-50)
    

    <ipython-input-53-c3fb264e2fe6> in curve_lookup(self, col, x, y)
        330                     for i in range(1,len(self.df[col])):
        331                         print(self.df[col].iloc[i-1],self.df[col].iloc[i])
    --> 332                         if y >= self.df[col].iloc[i-1] and y < self.df[col].iloc[i]:
        333 
        334                             result = self.df[col].iloc[i].index
    

    C:\Users\T3066SA\AppData\Local\Continuum\Anaconda3\lib\site-packages\pandas\core\generic.py in __nonzero__(self)
        890         raise ValueError("The truth value of a {0} is ambiguous. "
        891                          "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
    --> 892                          .format(self.__class__.__name__))
        893 
        894     __bool__ = __nonzero__
    

    ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().



```python

```
