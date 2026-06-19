print("\n41. Decision Tree")
age = int(input("Enter Age: "))
if age < 18:
    print("Child")
elif age < 60:
    print("Adult")
else:
    print("Senior Citizen")

print("\n42. Random forest")
print("\n vote Yes/No")
vote1 = input()
vote2 = input()
vote3 = input()
votes = [vote1, vote2, vote3]
if votes.count("Yes") > votes.count("No"):
    print("Final Prediction: Yes")
else:
    print("Final Prediction: No")

print("\n43. KNN")
import math
data = [(1, "A"),(2, "A"),(5, "B"),(6, "B")]
test = 3
distances = []
for point, label in data:
    d = abs(test - point)
    distances.append((d, label))
distances.sort()
k = 3
neighbors = distances[:k]
labels = [label for _, label in neighbors]
prediction = max(set(labels), key=labels.count)
print("Prediction:", prediction)

print("\n44. Naive Bayes")
import pandas as pd
# Load dataset
df = pd.read_csv(r"C:\internshiptask\spam.csv")
# Count classes
spam_count = len(df[df["Label"] == "Spam"])
notspam_count = len(df[df["Label"] == "NotSpam"])
total = len(df)
# Prior probabilities
p_spam = spam_count / total
p_notspam = notspam_count / total
print("P(Spam) =", p_spam)
print("P(NotSpam) =", p_notspam)
# Test message
message = input("Enter message: ").lower()
# Count occurrences of words
spam_score = p_spam
notspam_score = p_notspam
for word in message.split():
    spam_word_count = 0
    notspam_word_count = 0
    for i in range(len(df)):
        text = df.loc[i, "Message"].lower()
        if word in text:
            if df.loc[i, "Label"] == "Spam":
                spam_word_count += 1
            else:
                notspam_word_count += 1
    spam_score *= (spam_word_count + 1) / (spam_count + 2)
    notspam_score *= (notspam_word_count + 1) / (notspam_count + 2)
# Prediction
if spam_score > notspam_score:
    print("Prediction: Spam")
else:
    print("Prediction: Not Spam")

print("\n45. SVM")
x = float(input("Enter Value: "))
boundary = 5
if x > boundary:
    print("Class 1")
else:
    print("Class 0")

print("\n46. K-Means")
data = [1, 2, 3, 10, 11, 12]
cluster1 = []
cluster2 = []
c1 = 2
c2 = 11
for x in data:
    if abs(x - c1) < abs(x - c2):
        cluster1.append(x)
    else:
        cluster2.append(x)
print("Cluster 1:", cluster1)
print("Cluster 2:", cluster2)

print("\n47. Hierarchical clustering")
clusters = [[1], [2], [10], [11]]
while len(clusters) > 2:
    clusters[0].extend(clusters[1])
    clusters.pop(1)
print(clusters)

print("\n48. PCA")
data = [[1, 2],[2, 4],[3, 6]]
reduced = []
for row in data:
    reduced.append(row[0])
print(reduced)

print("\n49. House price prediction")
length=float(input("Length:"))
width=float(input("Width:"))
price=float(input("price per area"))
area=length*width
Total_price=area*price
print("House price:",Total_price)

print("\n50. Spam classifier")
message = input("Enter Message: ")
spam_words = ["free", "offer", "win", "prize"]
is_spam = False
for word in spam_words:
    if word in message.lower():
        is_spam = True
        break
if is_spam:
    print("Spam Message")
else:
    print("Not Spam")