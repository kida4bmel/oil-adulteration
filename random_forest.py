"""
Module for training RF models and/or doing hyperparameter tuning
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import montecarlo
import dataloading

class Datasplit:
    def __init__(self, data):
        self.data=data

    def get_data_from_csv(self):
        """
        Loads data from csv file
        """
        df = pd.read_csv(self.data)
        df = df.drop('Unnamed: 0', axis=1)
        return df
    
    def data_splitting(self, dataset):
        """
        Splits data for training models in balanced splits
        """
        datapoints_only = dataset.drop(["Label"],axis=1)
        labels = pd.DataFrame(dataset['Label'].values,columns=['Label'])
        datapoints_values= datapoints_only.values
        labels_values =labels.values

        X_train, X_test, y_train, y_test = train_test_split(datapoints_values, labels_values.flatten(), test_size=0.2, random_state=8, stratify=labels_values.flatten())
        return X_train, X_test, y_train, y_test
    
class HyperParameterTuning:
    def __init__(self, grid):
        self.grid = grid

    def hyperparameter_tuning(self, X_train, y_train):
        """
        tuning a baseline RF using Grid Search
        """
        self.estimator_tuning = RandomForestClassifier() # use a seed for getting always the same results from tuning
        param_search = GridSearchCV(estimator=self.estimator_tuning, param_grid=self.grid, n_jobs=-1, verbose=1)
        param_search.fit(X_train, y_train)
        best_params = param_search.best_params_
        return best_params

# use this if you know the parameters you want to use and do not need to do hyperparameter tuning
class RandomForest:
    def __init__(self, bootstrap, max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators, random_state, verbose):
        self.bootsrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split= min_samples_split
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose

    def train(self, X_train, y_train):
        """
        Training Rf with determined parameters and get training accuracy on train and test data
        """
        self.model = RandomForestClassifier(
            bootstrap=self.bootsrap,
            max_depth=self.max_depth,
            max_features= self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.model.fit(X_train,y_train )
        print('Acuracy train data: ', accuracy_score(y_train, self.model.predict(X_train))*100)
        print('Accuracy test data: ',accuracy_score(y_test, self.model.predict(X_test))*100)

    def save_model(self, filename):
        """
        saves model in pickle file
        """
        pickle.dump(self.model, open(filename, "wb"))
        print("Model saved")

        return filename
   
if __name__ == "__main__":
    num_sim= 100
    ratios = [[0.99, 0.01], [0.97, 0.03],[0.95, 0.05],[0.93, 0.07] ,[0.91, 0.09]   ] 
    thres1 = 0.05
    thres2 = 0.23
    path_to_csv = "yourdata.csv" #enter own path
    filename_model = "yourmodel" #enter own path

    #model parameters (use when not doing parameter tuning)
    model_parameters ={"bootstrap" :True, 
                 "max_depth":24, 
                 "max_features":None, 
                 "min_samples_leaf":7, 
                 "min_samples_split": 4, 
                 "n_estimators":150,
                 "random_state":8, 
                  "verbose":3} 
    
    # defining RF (just for training, not parameter tuning)
    rf1 = RandomForest(model_parameters["bootstrap"], 
                       model_parameters["max_depth"], 
                       model_parameters["max_features"], 
                       model_parameters["min_samples_leaf"],
                       model_parameters["min_samples_split"],
                       model_parameters["n_estimators"],
                       model_parameters["random_state"],
                       model_parameters["verbose"])
    
    # define a grid of parameters for hyperparameter tuning 
    random_grid = {'n_estimators': [100,150] ,
               "bootstrap":  [True],
               'max_features': [ None],
               'max_depth': [20,24],
                "min_samples_split": [2,4],
               "min_samples_leaf":[7,9]} 

    # Option 1: get previously generated data from csv
    data_csv = Datasplit(path_to_csv)
    dataset= data_csv.get_data_from_csv()
    X_train, X_test, y_train, y_test = data_csv.data_splitting(dataset)

    # Option 2: generate data anew
    # path_to_excel_co = 'CO.xlsx'
    # path_to_excel_so = 'SO.xlsx'
    # path_to_excel_realmix = "realmixtures.xlsx"
    # loaded_data_co = dataloading.LoadOilData(path_to_excel_co)
    # loaded_data_so = dataloading.LoadOilData(path_to_excel_so)
    # loaded_data_real =  dataloading.LoadOilData(path_to_excel_realmix)
    # co = loaded_data_co.get_dataframe()
    # co_label = loaded_data_co.return_label()
    # so =  loaded_data_so.get_dataframe()
    # so_label = loaded_data_so.return_label()
    # feats = loaded_data_co.return_features()
    # real = loaded_data_real.get_dataframe()
    # labelreal = loaded_data_real.return_label()
    # data_generation = montecarlo.MonteCarlo_simulator(co, so, co_label, so_label, feats)
    # dataset_classification = data_generation.all_steps_together(num_sim, ratios, labelreal, thres1, thres2)
    # data_generated = Datasplit(dataset_classification)
    # X_train, X_test, y_train, y_test = data_generated.data_splitting(dataset_classification)
    
    # parameter tuning
    #grid = HyperParameterTuning(random_grid)
    #params = grid.hyperparameter_tuning(X_train, y_train)
    #print(params)

    # training + saving model
    acc_train = rf1.train(X_train, y_train)
    rf1.save_model(filename_model)
