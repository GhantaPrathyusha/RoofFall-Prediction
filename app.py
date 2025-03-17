from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
import io
import base64

app = Flask(__name__)

data = pd.read_csv('Dataset.csv')
print(data.columns)

data.columns = data.columns.str.strip()

X = data[['Area of exposure (AE) sq. m', 'Depth of cover (DOC) m', 'RMR', 'Gallery Width', 'Inter-sectional Span (IS) m']]
y = data['Roof Fall Rate']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(eval_metric='rmse', random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model)],
    final_estimator=rf_model
)
stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)


def predict_distance(AE, DOC, RMR, GW, IS):
    input_data = np.array([[AE, DOC, RMR, GW, IS]])
    predicted_distance = stacking_model.predict(input_data)
    return predicted_distance[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            AE = float(request.form['AE'])
            DOC = float(request.form['DOC'])
            RMR = float(request.form['RMR'])
            GW = float(request.form['GW'])
            IS = float(request.form['IS'])
            
            result = predict_distance(AE, DOC, RMR, GW, IS)
            if result <= 1500:
                category_message = "The roof fall rate is Low."
            elif 1501 <= result <= 2500:
                category_message = "The roof fall rate is Medium."
            elif 2501 <= result <= 3500:
                category_message = "The roof fall rate is High."
            else:
                category_message = "The roof fall rate is Very High."

            return render_template('index.html', result=result, category_message=category_message)

        except ValueError:

            return render_template('index.html', result="Invalid input. Please enter valid numeric values.")

if __name__ == '__main__':
    app.run(debug=True)