from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import dataprep
import random_forest


class EvaluateModel:
    def __init__(self, filename_model):
        self.trained_model = pickle.load(open(filename_model, "rb"))

    def get_acuracy_test(self, y_test, X_test):
        y_pred = self.trained_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred )*100
        return accuracy, y_pred

    def get_classification_report_confusion_mmatrix(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        plt.figure(figsize=(10,6))
        plt.yticks(ticks =[0.5, 1.5, 2.5, 3.5,4.5,5.5,6.5,7.5,8.5,9.5] )
        report = sns.heatmap(report_df.T, annot=True, linewidths=5, linecolor="gray", cmap="Blues"  , fmt='g', annot_kws={"size":15} )
        plt.title('Classification Report')
        cm = confusion_matrix(y_test, y_pred, labels = self.trained_model.classes_)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.trained_model.classes_).plot()
        plt.title('Confusion Matrix')
        plt.show()


def load_real_mixtures(oil1, oil2, mix):
    actual_data = pd.concat([oil1, oil2, mix] )
    actual_datapoints_only = actual_data.drop(["Label"],axis=1)
    actual_labels = pd.DataFrame(actual_data['Label'].values,columns=['Label'])   
    X_test_actual = actual_datapoints_only.values
    y_test_actual =actual_labels.values.flatten()

    return actual_data, X_test_actual, y_test_actual

if __name__ == "__main__":

    filename_model = "model"
    #load model
    loaded_model = EvaluateModel(filename_model)

    #option1= evaluate model with simulated test data
    # path_to_csv =  "csvfile.csv"
    # data_csv = random_forest.Datasplit(path_to_csv)
    # dataset= data_csv.get_data_from_csv()
    # X_train, X_test, y_train, y_test = data_csv.data_splitting(dataset)
    # acc_test, y_pred = loaded_model.get_acuracy_test(y_test, X_test)
    # print(acc_test)
    # loaded_model.get_classification_report_confusion_mmatrix(y_test, y_pred)

    #option2= evaluate model with actual mixtures
    path_to_excel_co = 'CO.xlsx'
    path_to_excel_so = 'SO.xlsx'
    path_to_excel_realmix = "realmixtures.xlsx"
    loaded_data_co = dataprep.LoadOilData(path_to_excel_co)
    loaded_data_so = dataprep.LoadOilData(path_to_excel_so)
    loaded_data_real =  dataprep.LoadOilData(path_to_excel_realmix)
    co = loaded_data_co.get_dataframe()
    so = loaded_data_so.get_dataframe()
    actual = loaded_data_real.get_dataframe()
    actual_all_pd, X_test_real, y_test_real = load_real_mixtures(co, so,actual)
    #y_test_real_label = [label.replace("%","")for label in y_test_real] #use only when loading old generated csv
    acc_test, y_pred = loaded_model.get_acuracy_test(y_test_real, X_test_real)
    print(acc_test)
    loaded_model.get_classification_report_confusion_mmatrix(y_test_real, y_pred)


    