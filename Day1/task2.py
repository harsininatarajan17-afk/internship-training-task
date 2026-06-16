import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\Harsini\Downloads\iris.csv")
pd.DataFrame(df)
print(df)
print(df.info())
df2=df.dropna()
print(df2)
print(df.duplicated())
print(df.columns)
plt.scatter(df["sepal_length"], df["sepal_width"],color='hotpink')
plt.show()




