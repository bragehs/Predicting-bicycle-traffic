import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from Klasser.run_models import RunModels
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

komplett_data = pd.read_csv('komplett_data.csv', index_col=0, parse_dates=True)

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


#Fjern kommentarene her for å se korrelasjonsmatrise med de valgte variablene. 
#plt.figure(figsize=(10, 6)) 
#sn.heatmap(feature_engineering(komplett_data).corr(), annot= True)
#plt.show()


columns = ['Globalstraling','Solskinstid','Lufttemperatur', 'Vindretning', 'Vindstyrke', 'Vindkast'] 
X = feature_engineering(komplett_data.loc[:, columns])
y = komplett_data['Trafikkmengde']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle = False)


dummy = DummyRegressor(strategy='mean')

dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
print(f'\nDummy RMSE: {round(mean_squared_error(y_pred, y_test, squared = False), 2)}\n\n')




models = RunModels(X_train, X_test, y_train, y_test)

#finner beste modell for alle våre regressorer
#models.KNN()
#models.Linear()
#models.Tree()
#models.RFR()



#kjører beste modell på test data
models.run_best_model()

print('\n\nSaving model...')
models.save_model()
print('Saved to model.pkl')

#kjør den best modellen på 2023 værdata og lagre prediksjoner

print('\n\nPredicting for 2023 data...')

ny_data = pd.read_csv('2023.csv', index_col=0, parse_dates=True)

ny_X = feature_engineering(ny_data.loc[:, columns])
ny_y = ny_data['Trafikkmengde']

predictions = pd.DataFrame(columns= ['Dato', 'Tid', 'Prediction'])
predictions['Dato'] = ny_X.index.date
predictions['Tid'] = ny_X.index.time
preds = np.int64(models.load_best_model().predict(ny_X))
predictions['Prediction'] = preds
rmse = mean_squared_error(ny_y, preds, squared = False)
print(f'\n2023 RMSE:{rmse}\n')
#predictions.to_csv('predictions.csv')

print('Predictions saved to predictions.csv')