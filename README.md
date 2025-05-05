# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries

2. Load and analyse the dataset

3. Preprocess the data, convert the numerical to categorical data and encode this categorial codes to numeric codes using .cat.codes

4. Assign the input features and target variable

5. Define the functions sigmoid, loss, gradient_descent

6. Make predictions on new data and measure the accuracy

## Program & Output:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SRIRAM E
RegisterNumber:  212223040207
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Placement_Data.csv")
data.info()
```
![image](https://github.com/user-attachments/assets/32343f15-52cb-4066-9224-0ffc69f4a040)

```
data=data.drop(['sl_no','salary'],axis=1)
data
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes
```
![image](https://github.com/user-attachments/assets/584e6e1c-0d61-4c16-9f90-258743967086)

```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
```
![image](https://github.com/user-attachments/assets/564a1726-f82c-47c6-b773-bcbf0a5ee85a)

```
x=data.iloc[:,:-1].values
y=data.iloc[:,-1]
theta = np.random.randn(x.shape[1])
```
```
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
    
def loss(theta, x, y):
  h = sigmoid(x.dot(theta))
  return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```
```
def gradient_descent(theta, x, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(x.dot(theta))
    gradient = x.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta  
    
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)
    
def predict(theta, x):
  h = sigmoid(x.dot(theta))
  y_pred = np.where(h >= 0.5, 1, 0)
  return y_pred
```
```
y_pred = predict(theta, x)
accuracy=np.mean(y_pred.flatten()==y)
print("Acuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/aba30f4a-cf63-459d-b8d9-5a8ae1a97574)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/736acb77-86cb-4447-a7bf-878c64e3495c)

```
xnew=np.array([[0,0,0,5,5,2,3,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/8c5508c1-9873-450d-87c3-59cd37f0d487)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

