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
import matplotlib.pyplot as plt
import seaborn as sns
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

# Store results for plotting
mse_results = {}
r2_results = {}

# Train and evaluate models
print("Model performance evaluated using R² score and Mean Squared Error (MSE):\n")
for name, model in models.items():
    pipeline = Pipeline([('poly', PolynomialFeatures(degree=2)), ('model', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    mse_results[name] = mse
    r2_results[name] = r2
    
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")

# Create bar plots for MSE and R² comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MSE Bar Plot
sns.barplot(x=list(mse_results.keys()), y=list(mse_results.values()), ax=axes[0], palette='Blues_d')
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].set_ylabel('MSE')

# R² Score Bar Plot
sns.barplot(x=list(r2_results.keys()), y=list(r2_results.values()), ax=axes[1], palette='viridis')
axes[1].set_title('R² Score')
axes[1].set_ylabel('R² Score')

# Show plots
plt.tight_layout()
plt.show()

```

## Output:

Model performance evaluated using R² score and Mean Squared Error (MSE):
Ridge - MSE: 39201600.88, R²: 0.50
Lasso - MSE: 12616438.15, R²: 0.84
ElasticNet - MSE: 8666607.74, R²: 0.89
model = cd_fast.enet_coordinate_descent(
![image](https://github.com/user-attachments/assets/4c149fb2-d9ed-47b9-959e-703411382ce6)

## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
