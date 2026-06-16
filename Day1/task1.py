from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict
value = 6
prediction = model.predict([[value]])

print("Input:", value)
print("Predicted Output:", prediction[0])

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)