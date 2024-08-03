import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'Model Data.csv')


data = pd.read_csv(r'C:\Users\Rakesh Choudhury\OneDrive\Desktop\IBM Data Analytics\Model Data.csv')

# FEATURES--

features = data[['Projector %', 'Smart Class facility %', 'Digital Library %', 
                 'Computer Facility %', 'Internet Facility %', 'Playground %', 
                 "Functional Girls' Toilet%", "Functional Boys' Toilet %", 
                 'Functional Electricity %', 'Functional Drinking Water %', 
                 'Hand wash facility %']]

#TARGETS--

promotion_rate = data['Promotion Rate']
dropout_rate = data['Dropout Rate']

# Train and Test Data--

X_train, X_test, y_train_prom, y_test_prom = train_test_split(features, promotion_rate, test_size=0.2, random_state=42)
_, _, y_train_drop, y_test_drop = train_test_split(features, dropout_rate, test_size=0.2, random_state=42)

# Train the Model--

prom_model = RandomForestRegressor()
drop_model = RandomForestRegressor()

prom_model.fit(X_train, y_train_prom)
drop_model.fit(X_train, y_train_drop)

# Save the Model--
with open('prom_model.pkl', 'wb') as prom_file, open('drop_model.pkl', 'wb') as drop_file:
    pickle.dump(prom_model, prom_file)
    pickle.dump(drop_model, drop_file)
