import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
data = pd.read_csv("C:/Users/ACER/OneDrive/Desktop/Class/Python Class/Project/Mall_Data_3000_Entries-123.csv")

# First Five rows print
print("First 5 Rows:")
print(data.head())

print(data.info())
print(data.describe())

# Drop missing values
data = data.dropna()

# Columns Print 
print(data.columns)


# 1. Histogram (Distribution)
plt.figure()
data['Age Group'].hist()
plt.title("Histogram: Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 2. Scatter plot (Relation)
plt.scatter(data['Items Purchased'], data['Total Amount ($)'])
plt.xlabel("Items Purchased")
plt.ylabel("Total Amount ($)")
plt.title("Scatterplot: Items vs Total Spending")
plt.grid(True)
plt.show()

# 3. Box Plot (Outliers check)
sns.boxplot(x='Gender', y='Total Amount ($)', data=data)
plt.title("Box Plot: Spending by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Amount ($)")
plt.show()

# 4. Heatmap (Correlation)
corr = data[['Items Purchased', 'Unit Price ($)', 'Total Amount ($)', 'Discount (%)']].corr(numeric_only=True)
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap: Correlation Heatmap")
plt.show()

#Linear Regression
X = data[['Items Purchased']]
y = data['Total Amount ($)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
input_value = 5
input_data = pd.DataFrame({'Items Purchased': [input_value]})
predicted_amount = model.predict(input_data)
print(f"Items Purchased : {input_value}")
print(f"Estimated Total Amount ($) : {predicted_amount[0]:.2f}")


# Plot regression line
plt.scatter(data['Items Purchased'], data['Total Amount ($)'], color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Items Purchased")
plt.ylabel("Total Amount ($)")
plt.title("Regression: Items vs Total Spending")
plt.grid(True)
plt.legend()
plt.show()





