# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Vetrivel
RegisterNumber:  212225040487
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("First five row of dataset")
print(df.head())
print("Last five row of dataset")
print(df.tail())
x = df.iloc[:,:-1].values

y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("Predicted values")
print(y_pred)
print("Actual value")
print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
<img width="728" height="417" alt="Screenshot 2026-02-02 110151" src="https://github.com/user-attachments/assets/fe1e71af-2c54-426b-933f-0d6382ecc5f4" />
<img width="816" height="562" alt="Screenshot 2026-02-02 110233" src="https://github.com/user-attachments/assets/7c6d9b76-22c3-428f-bbc6-8ce22ebcc741" />
<img width="887" height="655" alt="Screenshot 2026-02-02 110245" src="https://github.com/user-attachments/assets/f5d2b6ad-cf2c-4455-8f42-cf69a73aaee6" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
