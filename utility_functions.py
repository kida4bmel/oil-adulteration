from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#function for plotting the confusion matrix
def plot_confusion_matrix_and_report(preds, targets):
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CO', 'SO','99:1', '97:3', '95:5', '93:7','91:9'])
    disp.plot() #cmap=plt.cm.Blues)
    sns.set_theme(font_scale=1.3)
    plt.show()
    report = classification_report(targets, preds, output_dict=True, target_names=['CO', 'SO', '99:1', '97:3', '95:5', '93:7', '91:9'] ) 
    report_df = pd.DataFrame(report).round(2)
    plt.figure(figsize=(10,6))
    sns.heatmap(report_df.T, annot=True, cmap="Blues"  , fmt='g', annot_kws={"size":15} , linewidths= 5, linecolor='grey', )
    sns.set_theme(font_scale=1.3)
    #plt.yticks( ['CO', 'SO', '99:1', '97:3', '95:5', '93:7', '91:9'])
    plt.yticks(ticks =  [0.5, 1.5, 2.5, 3.5 , 4.5 , 5.5, 6.5, 7.5, 8.5, 9.5] ,   labels=['CO', 'SO', '99:1', '97:3', '95:5', '93:7', '91:9', 'acc', 'mavg', 'wavg'] )
    plt.title('Classification Report')
    plt.show() 
def objective(trial):
    #hyperparameter die optimiert werden
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    hidden_dims = [trial.suggest_int(f'hidden_dim_{i}', 32, 128) for i in range(hidden_layers)]
    batch_size = trial.suggest_int('batch_size', 16, 64)
    label_column = 'Label'
    data_module = PandasDataModule(file_path, feature_columns=feature_columns, label_column=label_column, batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup()
    input_size = len(feature_columns)
    hidden_size = 128
    output_size = len(data_module.dataframe[label_column].unique())
    model = OilModel(input_size, hidden_size, output_size) 
    early_stopping = L.pytorch.callbacks.EarlyStopping(monitor="train_accuracy", mode="max", patience=40, verbose=True)
    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(save_top_k=1, monitor="train_accuracy", mode="max", verbose=False)
    trainer = L.Trainer(limit_train_batches=10, max_epochs=300, log_every_n_steps=5, logger=logger, callbacks=[early_stopping, model_checkpoint, ModelSummary(max_depth=-1)],  profiler="simple", devices=1)
    trainer.fit(model, data_module)
    best_score = None
    for callback in trainer.callbacks:
        # gets the best score from the earlystopping callback, which logs it.
        if hasattr(callback, 'best_score'):
            best_score = callback.best_score
            break
    return best_score            

def inference(model, unseen_csv_file):
    df_unseen = pd.read_csv(unseen_csv_file).fillna(0)
    features_unseen = df_unseen.values
    features_unseen_tensor = torch.tensor(features_unseen, dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(features_unseen_tensor).to(device)
        #predictions = torch.hstack(predictions)
        predicted_classes = torch.argmax(predictions, dim=1).to(device)
    return predicted_classes

list_column_names = ['PaOlPa', 'PaLiPa', 'PaOlSt', 'PaOlOl', 'PaLiSt', 'PaLiOl', 'PaLiLi',
       'StOlOl', 'OlOlOl', 'StLiOl', 'OlLiOl', 'LiLiOl', 'LiLiLi', ' 14:0',
       '16:0', '16:1Δ7', '16:1Δ9', '18:0', '18:1Δ9tr', '18:1Δ9', '18:1Δ11',
       '18:2Δ9t,12t', '18:2Δ9,12', '18:3Δ9,12,15', '20:0', '20:1Δ11', '22:0',
       '22:1Δ13', '20:5Δ5,8,11,14,17', '24:0', 'α-tocopherol', 'β-tocopherol',
       'γ-tocopherol', 'β-tocotrienol', 'Plastochromanol-8', 'γ-tocotrienol',
       'δ-tocopherol']

def get_dataframe (excel_file, sheet_name):
    df_oil = pd.read_excel(excel_file, sheet_name, header=3 )
    #df_oil = df_oil.drop(df_oil.columns[[15,16]] , axis=1) 
    #adding a last column with oil category
    sample_names = df_oil.iloc[:,0:1].to_numpy()
    samplenames = sample_names.flatten()
    label_column =[name[:-2] for name in samplenames]  
    label_column =[name.replace("-", "") for name in label_column] 
    label_column_pd= pd.DataFrame(label_column).rename(columns={0:"Label"} )
    df_oil["Label"]  = label_column_pd   
    return df_oil

def create_subsets(df_oil, label):
    list_column_names = ['PaOlPa', 'PaLiPa', 'PaOlSt', 'PaOlOl', 'PaLiSt', 'PaLiOl', 'PaLiLi',
       'StOlOl', 'OlOlOl', 'StLiOl', 'OlLiOl', 'LiLiOl', 'LiLiLi', ' 14:0',
       '16:0', '16:1Δ7', '16:1Δ9', '18:0', '18:1Δ9tr', '18:1Δ9', '18:1Δ11',
       '18:2Δ9t,12t', '18:2Δ9,12', '18:3Δ9,12,15', '20:0', '20:1Δ11', '22:0',
       '22:1Δ13', '20:5Δ5,8,11,14,17', '24:0', 'α-tocopherol', 'β-tocopherol',
       'γ-tocopherol', 'β-tocotrienol', 'Plastochromanol-8', 'γ-tocotrienol',
       'δ-tocopherol']
    oil  = df_oil[df_oil["Label"] == label] 
    oil_means= oil.loc[:,list_column_names[1] :list_column_names[-2]:2] 
    oil_sd = oil.loc[:,list_column_names[2]:list_column_names[-2]:2]
    return oil, oil_means, oil_sd