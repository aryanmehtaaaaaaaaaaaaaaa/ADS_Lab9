import pandas as pd
from flask import Flask, request, render_template
from model_files.inference import get_image_label

app = Flask(__name__)

@app.route("/")
def home():
    return """You redirect to following: 
            <br> 
            /form to access form 
            <br>
            /predict to access spam classifier
            <br>
            /classify to access digit classifier"""

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "GET":
        return render_template("form.html")

    name = request.form.get("name")
    mail = request.form.get("mail")
    pass_ = "".join(["*" for _ in range(len(request.form.get("pass")))])
    return render_template("form.html", name=name, mail=mail, pass_=pass_)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df["label"] = df["type"].map({"ham": 0, "spam": 1})
    X = df["text"]
    y = df["label"]

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import GridSearchCV

    cv = CountVectorizer()
    X = cv.fit_transform(X)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB(alpha = 0.001, force_alpha=True) 
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    message = request.form["message"]
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    return render_template("predict.html", prediction=my_prediction)


@app.route("/classify", methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        return render_template('classify.html')

    if request.method == 'POST':
        file = request.files['image_file']
        print(file)
        image = file.read()
        label = get_image_label(image_bytes=image)
        label = str(label)
        print(label)
        return render_template('classify.html', label=label,)

if __name__ == "__main__":
    app.run(debug=True)
