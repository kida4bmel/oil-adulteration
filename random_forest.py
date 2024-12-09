from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import montecarlo
import dataprep

class Datasplit:
    def __init__(self, data):
        self.data=data

    def get_data_from_csv(self):
        df = pd.read_csv(self.data)
        df = df.drop('Unnamed: 0', axis=1)
        return df
    
    def data_splitting(self, dataset):
        "Splits data for training models in balanced splits"
        datapoints_only = dataset.drop(["Label"],axis=1)
        labels = pd.DataFrame(dataset['Label'].values,columns=['Label'])
        datapoints_values= datapoints_only.values
        labels_values =labels.values

        X_train, X_test, y_train, y_test = train_test_split(datapoints_values, labels_values.flatten(), test_size=0.2, random_state=8, stratify=labels_values.flatten())
        return X_train, X_test, y_train, y_test

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
        pickle.dump(self.model, open(filename, "wb"))
        print("Model saved")

        return filename

    
if __name__ == "__main__":
    num_sim= 1000
    ratios = [[0.99, 0.01], [0.97, 0.03],[0.95, 0.05],[0.93, 0.07] ,[0.91, 0.09]   ] 
    thres1 = 0.05
    thres2 = 0.23
    path_to_csv = "csvfile.csv"
    filename_model = "model"

    #customizable model parameters
    model_parameters ={"bootstrap" :True, 
                 "max_depth":9, 
                 "max_features":None, 
                 "min_samples_leaf":3, 
                 "min_samples_split":6, 
                 "n_estimators":1500,
                 "random_state":8, 
                  "verbose":1} 
    
    rf1 = RandomForest(model_parameters["bootstrap"], 
                       model_parameters["max_depth"], 
                       model_parameters["max_features"], 
                       model_parameters["min_samples_leaf"],
                       model_parameters["min_samples_split"],
                       model_parameters["n_estimators"],
                       model_parameters["random_state"],
                       model_parameters["verbose"])

    # method1: get previously generated data from csv
    data_csv = Datasplit(path_to_csv)
    dataset= data_csv.get_data_from_csv()
    X_train, X_test, y_train, y_test = data_csv.data_splitting(dataset)

    # method2: generate data anew
    # path_to_excel_co = 'CO.xlsx'
    # path_to_excel_so = 'SO.xlsx'
    # path_to_excel_realmix = "realmixtures.xlsx"
    # loaded_data_co = dataprep.LoadOilData(path_to_excel_co)
    # loaded_data_so = dataprep.LoadOilData(path_to_excel_so)
    # loaded_data_real =  dataprep.LoadOilData(path_to_excel_realmix)
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
    
    acc_train = rf1.train(X_train, y_train)
    rf1.save_model(filename_model)

    
