from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
text = ["Bla bla bla -- bla bla-- lba ", "bla bla bla bla bla bla lba lba fasd"]
good =  [0,  1]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(text)
model = MultinomialNB()
model.fit(x,good)
user_input = input("Enter your text to predict: ")
x_input = vectorizer.transform([user_input])
prediction = model.predict(x_input)
print("Prediction:", "AI" if prediction[0] == 0 else "Human")
