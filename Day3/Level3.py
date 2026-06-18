import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

print("31. Linear Regression")
X1 = np.array([[1], [2], [3], [4], [5]])
y1 = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X1, y1)
print(model.predict([[6]]))

print("\n32. Multiple Regression")
X2 = np.array([[1, 2],[2, 3],[3, 4],[4, 5]])
y2 = np.array([3, 5, 7, 9])
model = LinearRegression()
model.fit(X2, y2)
print(model.predict([[5, 6]]))

print("\n33. Polynomial Regression")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X1)
model = LinearRegression()
model.fit(X_poly, y1)
print(model.predict(poly.transform([[5]])))

print("\n34. Logistic Regression")
model = LogisticRegression()
model.fit(X1, y1)
print(model.predict([[2.5]]))

print("\n35. Train/Test Split")
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
print("X & y train:",X_train,"\n",y_train)
print("X & y test:",X_test,"\n",y_test)

print("\n36. Evaluation Metrics")
y_true = [2, 4, 6]
y_pred = [2.1, 3.9, 6.2]
print(mean_squared_error(y_true, y_pred))

print("\n37. Confusion Matrix")
y_true = [1, 0, 1, 1]
y_pred = [1, 0, 0, 1]
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n38. Scaling")
scaler = StandardScaler()
print(scaler.fit_transform(X1))

print("\n39. Overfitting")
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("\n40. Cross Validation")
X = [[1], [2], [3],[4], [5], [6]]
y = [0, 0, 0,1, 1, 1]
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=2)
print(scores)
print("Average Accuracy:", scores.mean())