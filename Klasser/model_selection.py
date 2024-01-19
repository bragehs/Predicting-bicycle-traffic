from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import make_scorer, mean_squared_error


class ModelSelection:
    
    def __init__(self, model, scaler, params, imputer):
        
        self.model = model
        self.scaler = scaler
        self.params = params
        self.imputer = imputer
        self.best_model = None
        self.val_score = None
        self.best_params = None
        self.X_test_transformed = None
    
    
    def RMSE(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared = False)    

    def grid_search_pipe(self, X_train, y_train, pipeline=None):
        rmse_scorer = make_scorer(self.RMSE, greater_is_better=False)

        grid = GridSearchCV(pipeline, self.params, scoring= rmse_scorer,
                            cv=5, n_jobs=-1, verbose=3)
                            
        grid.fit(X_train, y_train)

        

        return grid.best_estimator_, abs(grid.best_score_), grid.best_params_
                                    #greater_is_better = False gir en negasjon av RMSE, derfor abs()
        
    def train(self, X_train, y_train):
        
        feature_selection = SelectKBest(score_func=mutual_info_regression, k=10)

        pipeline_steps = [
            ('imputer', self.imputer),
            ('scaler', self.scaler),
            ('feature_selection', feature_selection),
            ('model', self.model)
        ]     
        pipeline = Pipeline(pipeline_steps)
        

        best_model, best_score, best_params = self.grid_search_pipe(X_train, y_train, pipeline)
        
        model_name = type(self.model).__name__ 
        print(f'model: ({model_name})\nBest hyperparameters: {best_params}\nCV score: {best_score}')
            
        if (self.best_model is None) or best_score > self.best_model[1]:     
            self.best_model = (best_model, best_score)
            self.best_params = best_params
                
        return self.best_model