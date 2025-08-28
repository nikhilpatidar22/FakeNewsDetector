import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv(".\\FakeNewsNet.csv")


X = df[['title', 'source_domain', 'tweet_num']]
y = df['real']


preprocessor = ColumnTransformer(
    transformers=[
        ('title', TfidfVectorizer(max_features=5000), 'title'),
        ('domain', OneHotEncoder(handle_unknown='ignore'), ['source_domain']),
        ('tweets', StandardScaler(), ['tweet_num'])
    ]
)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model,'.\\FN_model.pkl')