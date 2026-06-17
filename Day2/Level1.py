print("1. Setup Python\nHello, World!")
print("\n2. Variables & Data Types")
name = "Hars"
age = 19
height = 5.8
print(name)
print(age)
print(height)
print("\n3. Lists, Tuples & Dictionary")
my_list = [10, 20, 30]
my_tuple = (1, 2, 3)
my_dict = {"rollno":42,"Name": "Hars", "Age":19}
print(my_list)
print(my_tuple)
print(my_dict)
print("\n4. Functions & Loops")
def square(n):
    return n * n
for i in range(1, 6):
    print(square(i))
print("\n5. File Handling")
with open("sample.txt", "w") as f:
    f.write("Hello Python")
with open("sample.txt", "r") as f:
    print(f.read())
print("\n6. NumPy Arrays")
import numpy as np
a = np.array([1, 2, 3, 4, 5])
print(a)
print(a * 2)
print("\n7. Pandas DataFrame")
import pandas as pd
d= {"Name": ["A", "B", "C"],"Marks": [80, 90, 70]}
pd.DataFrame(d)
print(d)
print("\n8. Load CSV")
df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
print(df)
print("\n9. Data Cleaning")
print(df.isnull().sum())
df2= df.dropna()
print(df2)
print("\n10. Filtering & Sorting")
filtered = df[df["sepal_length"] > 5]
print(filtered)
sorted_df = df.sort_values(by="sepal_length")
print(sorted_df)
print("\n11. Grouping")
group = df.groupby("species").mean(numeric_only=True)
print(group)
print("\n12. Merge Datasets")
df3= pd.DataFrame({"id": [1, 2, 3],"species": ["setosa", "versicolor", "virginica"]})
df4 = pd.DataFrame({"id": [1, 2, 3],"flower": ["A", "B", "C"]})
merged = pd.merge(df3, df4, on="id")
print(merged)
print("\n13. Statistics Basics")
print(df.describe())
print("Mean:")
print(df.mean(numeric_only=True))
print("Median:")
print(df.median(numeric_only=True))
print("\n14. Data Visualization")
import matplotlib.pyplot as plt
import seaborn as sns
# Load Iris dataset
df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
# Create subplot figure
plt.figure(figsize=(18, 12))
# 1. Line Plot
plt.subplot(3, 3, 1)
plt.plot(df["sepal_length"], color="hotpink")
plt.title("Line Plot")
# 2. Bar Plot
plt.subplot(3, 3, 2)
species_count = df["species"].value_counts()
plt.bar(species_count.index, species_count.values, color="hotpink")
plt.title("Bar Plot")
# 3. Pie Chart
species_count = df["species"].value_counts()
plt.subplot(3, 3, 3)
plt.pie(species_count.values,labels=species_count.index,autopct="%1.1f%%",colors=["hotpink", "pink", "lightgreen"])
plt.title("Pie Chart")# 4. Histogram
plt.subplot(3, 3, 4)
plt.hist(df["sepal_length"], bins=10, color="hotpink")
plt.title("Histogram")
# 5. Heatmap
plt.subplot(3, 3, 5)
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap="RdPu")
plt.title("Heatmap")
# 6. Density Plot
plt.subplot(3, 3, 6)
sns.kdeplot(df["petal_length"], fill=True, color="hotpink")
plt.title("Density Plot")
# 7. Contour Plot
plt.subplot(3, 3, 7)
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)
plt.contour(X, Y, Z, colors="hotpink")
plt.title("Contour Plot")
# 8. Box Plot
plt.subplot(3, 3, 8)
plt.boxplot(df["sepal_length"],patch_artist=True,boxprops=dict(facecolor="hotpink"))
plt.title("Box Plot")
plt.tight_layout()
plt.show()
print("\n15. Scatter Chart")
plt.scatter(df["sepal_length"], df["petal_length"])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot")
plt.show()
print("\n16. Correlation")
print(df.corr(numeric_only=True))
print("\n17. Export Data")
df.to_csv("iris_output.csv", index=False)
print("Data Exported Successfully")
print("\n18. Date & Time Handling")
from datetime import datetime
now = datetime.now()
print("Current Date and Time:", now)
print("Date:", now.strftime("%d-%m-%Y"))
print("Time:", now.strftime("%H:%M:%S"))
print("\n19. Script")
def load_data():
    df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
    print(df.head())
load_data()
print("\n20. Data Project")
import pandas as pd
import matplotlib.pyplot as plt
# Load Iris Dataset
df = pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
# Display first 5 rows
print("First 5 Rows:")
print(df.head())
# Display dataset information
print("\nDataset Information:")
print(df.info())
# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Count of each species
print("\nSpecies Count:")
print(df["species"].value_counts())
# Average values grouped by species
print("\nAverage Measurements by Species:")
print(df.groupby("species").mean(numeric_only=True))
# Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(df["sepal_length"],df["petal_length"],color="hotpink")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Sepal Length vs Petal Length")
plt.grid(True)
plt.show()
# Export dataset
df.to_csv("iris_output.csv", index=False)
print("\nData exported successfully as iris_output.csv")