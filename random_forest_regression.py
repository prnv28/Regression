# Random Forest Regression (RFR)

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Fitting RFR Model with Data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
print(y_pred)


# Visualising the RFR Result (with higher resolution curve)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (Decision tree Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
