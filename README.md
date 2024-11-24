# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.5</H3>
<H3>DATE: 16.09.24</H3>

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Import the required libraries.

2. **Load Dataset**:  
   Load the dataset into the environment.

3. **Data Preprocessing**:  
   Handle missing values and encode categorical variables.

4. **Define Features and Target**:  
   Split the dataset into features (X) and the target variable (y).

5. **Create Polynomial Features**:  
   Generate polynomial features from the data.

6. **Set Up Pipelines**:  
   Create pipelines for Ridge, Lasso, and ElasticNet models.

7. **Train Models**:  
   Fit each model to the training data.

8. **Evaluate Model Performance**:  
   Assess performance using the R² score and Mean Squared Error (MSE).

9. **Compare Results**:  
   Compare the performance of the models.

## Program:
```py
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")
data = pd.get_dummies(data.drop(['CarName', 'car_ID'], axis=1), drop_first=True)

# Define features and target
X = data.drop('price', axis=1)
y = data['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline([('poly', PolynomialFeatures(degree=2)), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(f"{name} - MSE: {mean_squared_error(y_test, preds):.2f}, R²: {r2_score(y_test, preds):.2f}")

```

## Output:

model = cd_fast.enet_coordinate_descent(

Ridge - Mean Squared Error: 39011712.54, R² Score: 0.51

Lasso - Mean Squared Error: 12616438.15, R² Score: 0.84

ElasticNet - Mean Squared Error: 8666607.74, R² Score: 0.89)

<img width="1197" alt="Screenshot 2024-10-06 at 8 58 51 PM" src="https://github.com/user-attachments/assets/bfeebd0c-c84d-4dce-9c38-182990f46973">


## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
