from flask import Flask, render_template, request
import os
import pickle

app = Flask(__name__, template_folder='.')
model_path = os.path.dirname(os.path.abspath(__file__))
prom_model = pickle.load(open(os.path.join(model_path, 'prom_model.pkl'), 'rb'))
drop_model = pickle.load(open(os.path.join(model_path, 'drop_model.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        fields = ['projector', 'smart_class', 'digital_library', 'computer_facility', 'internet_facility', 
                  'playground', 'girls_toilet', 'boys_toilet', 'electricity', 'drinking_water', 'hand_wash']
        
        for field in fields:
            value = request.form.get(field)
            if not value:
                return render_template('index.html', error_message=f"Error: {field} cannot be empty. Please provide a valid value.", request=request)
            features.append(float(value))

        prediction_promotion = prom_model.predict([features])[0]
        prediction_dropout = drop_model.predict([features])[0]

        prediction_promotion = f"{prediction_promotion:.3f}"
        prediction_dropout = f"{prediction_dropout:.3f}"

        return render_template('index.html', prediction_promotion=prediction_promotion, prediction_dropout=prediction_dropout, request=request)
    except ValueError as e:
        return render_template('index.html', error_message=f"Error: Invalid input. Please provide numeric values for all fields. Details: {str(e)}", request=request)
    except KeyError as e:
        return render_template('index.html', error_message=f"Missing data for {str(e)}. Please check the form and try again.", request=request)

if __name__ == '__main__':
    app.run(debug=True)
