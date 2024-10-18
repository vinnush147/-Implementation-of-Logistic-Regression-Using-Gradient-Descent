
# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a Pandas DataFrame, drop unnecessary columns, and encode categorical variables into numerical values.
2. Define the sigmoid() function to map any real-valued number to a value between 0 and 1 using the logistic function.
3. Implement the loss() function to calculate the logistic loss (log loss), which measures how well the logistic model fits the data.
4. Perform gradient descent using the gradient_descent() function to iteratively update the model’s parameters (theta). The function minimizes the loss by adjusting the parameters over several iterations.
5. Use the predict() function to compute predictions for both training data and new input data. Calculate the accuracy by comparing predicted values to actual values and print the model's accuracy and predictions.


## Program:

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vinnush Kumar L S
RegisterNumber: 212223230244
```
```
from google.colab import drive
drive.mount('/content/gdrive')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Read the Dataset:
```
a=pd.read_csv('/content/Placement_Data_Full_Class (1).csv')
a
```
## Output:
![image](https://github.com/user-attachments/assets/cb81a177-2b4d-4139-b8b3-87896d323419)
## Info :
```
a.head()
a.tail()
a.info()
```

## Output:

![Screenshot 2024-10-16 094514](https://github.com/user-attachments/assets/dc2d6eb9-e900-4c03-bf7a-6107d63e72ab)
![Screenshot 2024-10-16 094655](https://github.com/user-attachments/assets/fdbc132c-9454-4142-b5c2-2d6fef972b89)
![image](https://github.com/user-attachments/assets/9ad47c29-1ea9-4738-a858-c26db8ed6da9)

## Drop unnecessary columns
```
a=a.drop(['sl_no'],axis=1)
a
```
## Output:
![Screenshot 2024-10-16 094934](https://github.com/user-attachments/assets/689b4970-82a2-48cb-80ad-f0f8742c1869)

## Encoding Categorical Variables:
```
a['gender']=a['gender'].astype('category')
a['ssc_b']=a['ssc_b'].astype('category')
a['hsc_b']=a['hsc_b'].astype('category')
a['hsc_s']=a['hsc_s'].astype('category')
a['degree_t']=a['degree_t'].astype('category')
a['workex']=a['workex'].astype('category')
a['specialisation']=a['specialisation'].astype('category')
a['status']=a['status'].astype('category')
a.info()

a['gender']=a['gender'].cat.codes
a['ssc_b']=a['ssc_b'].cat.codes
a['hsc_b']=a['hsc_b'].cat.codes
a['hsc_s']=a['hsc_s'].cat.codes
a['degree_t']=a['degree_t'].cat.codes
a['workex']=a['workex'].cat.codes
a['specialisation']=a['specialisation'].cat.codes
a['status']=a['status'].cat.codes
a.info()
```
## Output:
![Screenshot 2024-10-16 095131](https://github.com/user-attachments/assets/492de0b1-f138-4ece-993d-fdaf91dd1105)
![image](https://github.com/user-attachments/assets/c6a67e9c-c407-4c35-a9f6-196cccea0570)

## Splitting Data:
```
x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values
```
## Gradient Descent:
```
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient=X.T.dot(h-y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5 , 1,0)
  return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print(accuracy*100)
print(y_pred)
```
## Output :
![Screenshot 2024-10-16 114214](https://github.com/user-attachments/assets/a0cb9c87-e2ec-4e8d-8e96-9dd49ee9eee6)

## Prediction and Evaluation:
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_predn=predict(theta,xnew)
print(y_predn)
print(theta)
```
## Output:
![Screenshot 2024-10-16 114403](https://github.com/user-attachments/assets/ab805222-996e-473e-bccc-d277b6790d04)
![image](https://github.com/user-attachments/assets/969a5e07-9dc9-4f06-8fb9-2b93aa1766c9)






## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

