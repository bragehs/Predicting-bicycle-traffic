import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from flask import Flask, request, render_template
from waitress import serve

#samme feature_engineering funksjon for å variabelutvinne fra input data
def feature_engineering(X):
    X_copy = X.copy()
    
    X_copy['Time'] = X.index.hour

    X_copy['Måned'] = X.index.month

    X_copy['Dag'] = X.index.weekday + 1
    
    #Gjennomsnittstråling siste 3 timene, prøver å ta variert vær med i betraknting 
    X_copy['Gjennomsnittstråling siste 3 timene'] = X['Globalstraling'].rolling(window=3).mean() 

    #Hvor mye solen har skinnet en dag frem til et tidspunkt, hvis solen har vært ute vil mulig folk sykle
    X_copy['Soltid til nå'] = X.groupby(X.index.date)['Solskinstid'].cumsum().values

    #Prøver å lage en relasjon mellom temperatur og soltid, kreves begge deler for ekstra sykler
    X_copy['Soltid * Lufttemperatur'] = X['Lufttemperatur'] * X['Solskinstid']

    return X_copy

app = Flask(__name__)


model = pickle.load(open('/Users/bragehs/Documents/inf161/prosjekt1/Complete/model.pkl', 'rb'))
print((model))

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = dict(request.form)

    columns = ['Globalstraling', 'Solskinstid', 'Lufttemperatur','Vindretning', 'Vindstyrke']
    

    input_date = features.get('Dato', '')
    input_time = features.get('Tid', '')


    try:
    
        parsed_date = datetime.strptime(input_date, '%Y-%m-%d')
        parsed_time = datetime.strptime(input_time, '%H:%M')
    except ValueError:
        return render_template('./index.html', prediction_text='Invalid date or time format. NB! mellomrom etter inputen vil gi feilmelding')


    parsed_datetime = parsed_date.replace(
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=0,
        microsecond=0
    )


    #Hent ut trengte kolonner fra parsed_date
    features['Dag'] = parsed_date.weekday() + 1 
    features['Måned'] = parsed_date.month
    features['Time'] = parsed_date.hour
 
    

    #håndter ingen input
    def to_numeric(key, value):
        try:
            return float(value)
        except:
            return np.nan

    features = {key: to_numeric(key, value) for key, value in features.items()}

    features['Dato'] = pd.to_datetime(parsed_datetime)

    #lag og print dataframe med input verdiene
    features_df = feature_engineering(pd.DataFrame(features, index= [features['Dato']]).loc[:, columns])
    print(features_df)

    # prediker
    prediction = int((model.predict(features_df)[0]))
    print(prediction)

    # formater output
    return render_template('./index.html',
                           prediction_text='Predikert mengde sykler: {}'.format(prediction))

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
