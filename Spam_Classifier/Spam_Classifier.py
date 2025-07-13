import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# The dataset has two useful columns: 'v1' (label) and 'v2' (message)
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Converting the labels to binary values — ham: 0, spam: 1
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])  # Features
y = data['label_num']                          # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    print(f"\nMessage: {msg}")
    print("Prediction:", "Spam" if prediction == 1 else "Ham")

# Test on a few custom messages
predict_message("Congratulations! You've won a free iPhone. Claim now.")
predict_message("Hey, are we still meeting at 5?")
