print("\n51. Customer segmentation")
customers = [1000, 1200, 1500, 5000, 5500, 6000]
low = []
high = []
for amount in customers:
    if amount < 3000:
        low.append(amount)
    else:
        high.append(amount)
print("Low Spending Customers:", low)
print("High Spending Customers:", high)

print("\n52. Sales prediction")
sales = [100, 200, 300, 400]
average = sum(sales) / len(sales)
print("Predicted Next Month Sales:", average)

print("\n53. Fraud detection")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Dataset
amount = [[500], [1200], [8000], [60000], [75000], [300], [45000], [90000]]
fraud = [0, 0, 0, 1, 1, 0, 0, 1]

# Pie Chart
normal = fraud.count(0)
fraud_count = fraud.count(1)
plt.pie([normal, fraud_count],labels=["Normal", "Fraud"],autopct="%1.1f%%",colors=["lightgreen", "lightpink"])
plt.title("Fraud Detection")
plt.show()
# Train Model
model = DecisionTreeClassifier()
model.fit(amount, fraud)
# Prediction
amt = float(input("Enter Transaction Amount: "))
result = model.predict([[amt]])
if result[0] == 1:
    print("Fraud Transaction")
else:
    print("Normal Transaction")

print("\n54. Model comparison")
model1_accuracy = float(input("Model1 accuracy:"))
model2_accuracy = float(input("Model2 accuracy:"))
if model1_accuracy > model2_accuracy:
    print("Model 1 is Better")
else:
    print("Model 2 is Better")

print("\n55. Hyperparameter tuning")
depths = [2, 3, 4, 5]
best_accuracy = 0
best_depth = 0
for depth in depths:
    accuracy = depth * 10
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
print("Best Depth:", best_depth)
print("Best Accuracy:", best_accuracy)

print("\n56. Pipelines")
data = [10, 20, 30, 40]
# Step 1: Scaling
scaled = []
for x in data:
    scaled.append(x / 10)
# Step 2: Prediction
predictions = []
for x in scaled:
    predictions.append(x * 2)
print(predictions)

print("\n57. Save/load model")
import pickle
model = {"slope": 2,"intercept": 5}
# Save
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
# Load
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
print(loaded_model)

print("\n58. Feature engineering")
import matplotlib.pyplot as plt

age = [18, 19, 20, 21, 22]
marks = [70, 75, 80, 85, 90]

# New Feature
age_square = []
for i in age:
    age_square.append(i ** 2)
print("Age:", age)
print("Age Square:", age_square)
# Pie Chart
plt.pie(marks,labels=age,autopct="%1.1f%%",colors=["hotpink","pink","lightpink","deeppink","violet"])
plt.title("Marks Distribution by Age")
plt.show()

print("\n59. Imbalanced data")
import matplotlib.pyplot as plt
# Dataset
data = [0,0,0,0,0,0,0,0,1,1]
majority = data.count(0)
minority = data.count(1)
print("Majority Class:", majority)
print("Minority Class:", minority)
# Pie Chart
plt.pie([majority, minority],labels=["Class 0", "Class 1"],autopct="%1.1f%%",colors=["hotpink", "lightpink"])
plt.title("Imbalanced Data")
plt.show()

print("\n60. ML project Programs")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
df = pd.read_csv(r"C:\internshiptask\salary.csv")
# Features and Target
X = df[["Experience"]]
y = df["Salary"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict test data
y_pred = model.predict(X_test)
# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
# User input
exp = float(input("Enter Years of Experience: "))
salary = model.predict([[exp]])
print("Predicted Salary:", salary[0])
import matplotlib.pyplot as plt
# Pie Chart
plt.figure(figsize=(6,6))
plt.pie(df["Salary"],labels=df["Experience"],autopct="%1.1f%%")
plt.title("Salary Distribution by Experience")
plt.show()