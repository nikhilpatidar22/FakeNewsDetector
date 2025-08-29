from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load(".\\FN_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
       
        title = request.form["title"]
        tweet_num = request.form["tweet_num"]
        source_domain = request.form["source_domain"]

       
        input_df = pd.DataFrame([{
            "title": title,
            "tweet_num": int(tweet_num),
            "source_domain": source_domain
        }])

        
        pred = model.predict(input_df)[0]
        result = "✅ Real News" if pred == 1 else "❌ Fake News"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
