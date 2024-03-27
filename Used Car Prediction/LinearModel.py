from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Mapping categorical variables to numerical values
Fuel_Type = {'CNG':0,'Petrol':1,'LPG':2,'Diesel':3}
Owner_Type = {'First':0,'Second':1,'Third':2,'Forth & Above':3}
Transmission = {'Manual':0,'Automatic':1}
df['Fuel_Type'] = df['Fuel_Type'].map(Fuel_Type)
df['Owner_Type'] = df['Owner_Type'].map(Owner_Type)
df['Transmission'] = df['Transmission'].map(Transmission)

# Dropping rows with missing values
df = df.dropna()

# Define features (X) and target (Y)
Y = df['Price'].values
X = df.drop('Price',axis=1).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared (R2) score to evaluate the model
r2 = r2_score(y_test, y_pred)

# Convert R2 score to percentage for better interpretation
accuracy_percentage = r2 * 100

# Print the accuracy (R2) in percentage
print("Accuracy (R2) in Percentage:", accuracy_percentage)
