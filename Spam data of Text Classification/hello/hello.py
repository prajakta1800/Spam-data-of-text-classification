import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pylab as plt

df = pd.read_csv('spam_dataset.csv')
print(df)
df.info()
print("dataset shape :",df.shape)
print("daatset columns :",df.columns)

print("spliting dataset into training and testing ")
X_train, X_test, y_train, y_test = train_test_split(df['message_content'], df['is_spam'], test_size=0.2, random_state=42)
print(" data split ")
print("trainig set shape :",X_train.shape,y_train.shape)
print("testing set shape :",X_test.shape,y_test.shape)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("data transformed")

print(" vectorizer trainig set shape : ",X_train_vectorized.shape)
print(" vectorizer testing set shape : ",X_test_vectorized.shape)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
print("classifier trained")

y_pred = clf.predict(X_test_vectorized)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("confusion matrix :")
conf_mat=confusion_matrix(y_test,y_pred)
print(conf_mat)
TN,FP,FN,TP =conf_mat.ravel()
print("true negatives: ",TN)
print("false positives: ",FP)
print("false negatives: ",FN)
print("true positives: ",TP)

plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True,cmap="Blues")
plt.xlabel("predicted  message_content ")
plt.ylabel("true  message_content")
plt.title("confusion matrix")
plt.show()






