from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)
model = CatBoostClassifier()
model.load_model('trained_model.cbm')


@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    df = pd.DataFrame(content, index=[0])
    df = df.drop(['PassengerId'], axis=1)
    print(df)
    result = model.predict_proba(df)
    result = {'Survived_Probability': result[0, 1]}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
