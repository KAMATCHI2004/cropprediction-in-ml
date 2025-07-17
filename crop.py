import pandas

data = pandas.read_csv("crop.csv")
data['label'].unique()
data.info()
data.describe()
data.head(5)
data.tail(5)

data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()


data.duplicated().sum()
data.drop_duplicates(inplace=True)

from sklearn.preprocessing import LabelEncoder
columns_to_encode=["label"]
label_encoder=LabelEncoder()
for col in columns_to_encode:
    data[col]=label_encoder.fit_transform(data[col])
data.info()

X=data.drop(columns="label")
print(X)
Y=data[["label"]]
#from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=10,test_size=0.20)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
data.shape



# Initialize and train the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=70)
forest.fit(x_train, y_train)


#Predict on the test set
pred = forest.predict(x_test)


from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
pred=tree.predict(x_test)

# Calculate and print accuracy
accuracy = f"{round(accuracy_score(y_test, pred) * 100, 2)}%"
print(f"Accuracy: {accuracy}")

accuracy = accuracy_score(y_test, pred) * 100
print(f"Accuracy: {accuracy:.2f}%")



# Generate and print the confusion matrix
cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(cm)


import pickle as pk
with open("model.pkl", "wb") as file:
    pk.dump(forest,file)
