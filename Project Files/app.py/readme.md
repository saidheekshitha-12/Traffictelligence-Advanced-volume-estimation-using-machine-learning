from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)
model = pickle.load(open('C:/Users/Ramesh/Downloads/model(3).pkl', 'rb'))
scale = pickle.load(open('C:/Users/Ramesh/Downloads/scale(2).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Get values from form input
    input_features = [float(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]

    # Define feature column names
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
             'hours', 'minutes', 'seconds']

    # Convert to DataFrame
    data = pd.DataFrame(feature_values, columns=names)

    # Scale using pre-fitted scaler
    data_scaled = scale.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    # Return result to webpage
    text = "Estimated Traffic Volume is: "
    return render_template("index.html", prediction_text=text + str(int(prediction[0])))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True)  
