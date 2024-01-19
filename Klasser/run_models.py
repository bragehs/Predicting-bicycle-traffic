import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from Klasser.model_selection import ModelSelection


class RunModels:
    def __init__(self, X_train, X_test, y_train, y_test, 
    models = ['KNN', 'DTR', 'RDG', 'RFR'], scaler = StandardScaler(), imputer = KNNImputer()):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = models
        self.scaler = scaler
        self.imputer = imputer
        self.y_pred = None
        self.best_model = None
        self.pipe = None
        self.seed = 42
    
    def Linear(self, alpha = [1, 2, 10]):
        model = Ridge()
        params = {
            'model__alpha': alpha
        }
        rdg = ModelSelection(model, self.scaler, params, self.imputer)
        
        rdg.train(self.X_train, self.y_train)

      
        with open('RDG.pickle', 'wb') as file:
            pickle.dump(rdg, file)
        
    def KNN(self, n_neighbors = [5, 13, 50], p = [1, 2], weights = ['uniform', 'distance']):
        model = KNeighborsRegressor()
        params = {
            'model__n_neighbors': n_neighbors,
            'model__p': p,
            'model__weights': weights
        }
        knn = ModelSelection(model, self.scaler, params, self.imputer)
        
        knn.train(self.X_train, self.y_train)

        with open('KNN.pickle', 'wb') as file:
            pickle.dump(knn, file)
        


   

    def Tree(self, min_samples_leaf = [2,6,10] , max_features = ["log2","sqrt",None]):
        model = DecisionTreeRegressor(random_state= self.seed)
        params = {
            'model__min_samples_leaf': min_samples_leaf,
            'model__max_features': max_features
            
        }

        dtr = ModelSelection(model, self.scaler, params, self.imputer)

        dtr.train(self.X_train, self.y_train)

       
        with open('DTR.pickle', 'wb') as file:
            pickle.dump(dtr, file)


    def RFR(self, n_estimators = [250]):

        model = RandomForestRegressor(random_state= self.seed)
        params = {
            'model__n_estimators': n_estimators,
        }

        rfr = ModelSelection(model, self.scaler, params, self.imputer)


        rfr.train(self.X_train, self.y_train)


        with open('RFR.pickle', 'wb') as file:
            pickle.dump(rfr, file)

    def run_best_model(self):
        best_accuracy = float('inf')
        output = []
        for i in self.models:
            data = pickle.load(open(i + '.pickle', 'rb'))
            score = data.best_model[1]
            output.append([i, score])
            if score < best_accuracy:
                best_accuracy = score
                self.best_model = i
                params = data.best_params
            
        for i, j in output:
            print(f'Model: {i}\tScore: {round(j, 2)}\n')

        print(f'\nOur best model: {self.best_model}\nHyperparameters: {params}\n')
        
        print('\nRunning best model...')
        
        with open(self.best_model + '.pickle', 'rb') as file:
            loaded_model = pickle.load(file)

        y_pred = loaded_model.best_model[0].predict(self.X_test)
        self.y_pred = y_pred

        print(f'Our best model´s RMSE on test data: {mean_squared_error(self.y_test, y_pred,  squared = False)}')


    def load_best_model(self):
        with open(self.best_model + '.pickle', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model.best_model[0]

    def save_model(self):
        # lagre modell
        model = self.load_best_model()
        with open('best_model.pickle', 'wb') as f:
            pickle.dump(model, f)
        