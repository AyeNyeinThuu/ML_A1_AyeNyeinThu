from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('predicting_car_price.model','rb'))  # Ensure this file contains only the trained model

# Load the scaler
scaler = pickle.load(open('scaler.pkl','rb'))  # Ensure this file contains only the fitted scaler

# Define the brand encoding map
brand_encoded_map = {
    'Ambassador': 20, 'Ashok': 27, 'Audi': 10, 'BMW': 11, 'Chevrolet': 29,
    'Daewoo': 9, 'Datsun': 26, 'Fiat': 19, 'Force': 28, 'Ford': 4,
    'Honda': 7, 'Hyundai': 6, 'Isuzu': 14, 'Jaguar': 21, 'Jeep': 22,
    'Kia': 2, 'Land': 30, 'Lexus': 8, 'MG': 0, 'Mahindra': 1,
    'Maruti': 12, 'Mercedes-Benz': 24, 'Mitsubishi': 15, 'Nissan': 5,
    'Opel': 16, 'Peugot': 3, 'Renault': 13, 'Skoda': 18, 'Tata': 23,
    'Toyota': 17, 'Volkswagen': 25, 'Volvo': 31
}

# Transmission encoding
transmission_mapping = {'Manual': 1, 'Automatic': 0}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        brand = request.form['brand'].strip()
        year = int(request.form['year'])
        transmission = request.form['transmission'].strip()
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])

        # Debugging inputs
        print(f"Inputs received - Brand: {brand}, Year: {year}, Transmission: {transmission}, Engine: {engine}, Max Power: {max_power}")

        # Validate and encode inputs
        if brand not in brand_encoded_map:
            error_message = f"Brand '{brand}' is not recognized."
            return render_template('result.html', error=error_message)
        brand_encoded = brand_encoded_map[brand]

        if transmission not in transmission_mapping:
            error_message = 'Invalid transmission type selected.'
            return render_template('result.html', error=error_message)
        transmission_encoded = transmission_mapping[transmission]

        # Prepare the input features
        features = np.array([[brand_encoded, year, transmission_encoded, engine, max_power]])
        print(f"Features before scaling: {features}")

        # Scale the features
        features_scaled = scaler.transform(features)
        print(f"Features after scaling: {features_scaled}")

        # Predict the log-transformed price
        predicted_log_price = model.predict(features_scaled)
        print(f"Predicted log price: {predicted_log_price}")

        # Convert to original scale
        predicted_price = np.exp(predicted_log_price[0])
        print(f"Predicted price: {predicted_price}")

        # Return the prediction in an HTML page
        return render_template(
            'result.html',
            brand=brand,
            year=year,
            transmission=transmission,
            engine=engine,
            max_power=max_power,
            predicted_price=round(predicted_price, 2)
        )

    except Exception as e:
        error_message = str(e)
        return render_template('result.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
