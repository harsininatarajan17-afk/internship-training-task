import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\Harsini\Downloads\data.csv")
print("Student Dataset:")
print(df)
X = df[["Hours"]]      
y = df["Score"]        
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy (R² Score):", accuracy)
hours = float(input("\nEnter study hours: "))
new_data = pd.DataFrame({"Hours": [hours]})
predicted_score = model.predict(new_data)
print(f"Predicted Score: {predicted_score[0]:.2f}")