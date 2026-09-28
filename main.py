from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
text = ["Bla bla bla -- bla bla-- lba ", "bla bla bla bla bla bla lba lba fasd", "The algorithm — while complex — produces reliable results.", "It works, but not always", "The algorithm — while complex — produces reliable results.", "The fix is simple, just restart it.", "omg i just spilled coffee on my keyboard and now the s key is stuck lmaooo", "cant believe i stayed up till 3am watching that show again, i have no self control", "bro the wifi went out right when i was about to submit my homework i literally wanted to cry", "just ate leftover pizza for breakfast and i regret nothing", "my cat knocked my water bottle off the desk for the 5th time today. i think she hates me", "tried to parallel park today and it took me 4 attempts. the guy behind me was NOT happy", "why does monday always feel like it comes 3 days early", "forgot my headphones at home today and had to sit through the whole bus ride without music. felt like a punishment", "made pasta for dinner, ate half the pot standing over the stove. no plates needed", "my phone died at 1% right when i needed maps. classic"]
good = [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(text)
model = MultinomialNB()
model.fit(x,good)
user_input = input("Enter your text to predict: ")
x_input = vectorizer.transform([user_input])
prediction = model.predict(x_input)
print("Prediction:", "AI" if prediction[0] == 0 else "Human")
