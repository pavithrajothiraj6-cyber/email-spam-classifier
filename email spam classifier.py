import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB



emails = [
    "Congratulations! You won a free prize. Click here now",
    "You have won a lottery of 10 lakh rupees",
    "Claim your free gift now",
    "URGENT! You have won a cash prize",
    "Get free money by clicking this link",
    "Congratulations, you are selected for a free iPhone",
    "Win a free vacation today",
    "You have been selected for a lucky draw",
    "Free offer! Claim your reward now",
    "Click here to get your free coupon",

    "Hi, how are you?",
    "Can we meet tomorrow?",
    "Please send me the project report",
    "Your meeting is scheduled for tomorrow",
    "Hi, please call me when you are free",
    "The assignment is completed",
    "Don't forget to attend the class",
    "Can you send me the notes?",
    "Thank you for your email",
    "Let's discuss the project today"
]

labels = [
    "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam", "spam",

    "ham", "ham", "ham", "ham", "ham",
    "ham", "ham", "ham", "ham", "ham"
]




os.makedirs("models", exist_ok=True)




tfidf = TfidfVectorizer()

X = tfidf.fit_transform(emails)



model = MultinomialNB()

model.fit(X, labels)

with open("models/tfidf.pkl", "wb") as file:
    pickle.dump(tfidf, file)

with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
email = input("\nEnter Email: ")
vector = tfidf.transform([email])
prediction = model.predict(vector)[0]
if prediction == "spam":
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM")