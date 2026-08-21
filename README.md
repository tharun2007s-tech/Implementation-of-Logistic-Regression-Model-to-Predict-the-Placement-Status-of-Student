# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start
2. Import the required libraries: Pandas, NumPy, Matplotlib, and Logistic Regression.
3. Load the student placement dataset using pd.read_csv().
4. Remove the salary column because salary is known only after placement.
5. Convert categorical variables into numerical values using One-Hot Encoding with pd.get_dummies().
6. Separate the dataset into:
7. X → input features
8. y → placement status (status_Placed)
9. Split the dataset into training data (80%) and testing data (20%).
10. Create a Logistic Regression model.
11. Train the model using the training data.
12. Predict the placement status of the test data.
13. Calculate the accuracy of the model.
14. Display the prediction results.
15. Stop

## Program:

# Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: Jegan P
# RegisterNumber:  212225240061
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load data
data = pd.read_csv("Placement_Data (1).csv")

#Remove salary because it is not known before placement
data = data.drop("salary", axis=1)

#Convert categorical data into numerical data
data = pd.get_dummies(data, drop_first=True)

#Features and target
X = data.drop("status_Placed", axis=1)
y = data["status_Placed"]

#Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Accuracy
print("Accuracy:", model.score(X_test, y_test))

#Predict a new student
prediction = model.predict(X_test)

print("Predicted Placement Status:")
print(prediction)
```
## Output:

<img width="675" height="128" alt="image" src="https://github.com/user-attachments/assets/e93a4b5d-ccd5-487a-a5e6-189294c259bf" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
