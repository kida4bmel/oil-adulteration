import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics
import lightning as L
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import lightning as L
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
torch.set_float32_matmul_precision('high')
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics import Metric
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error as mae 
logger = TensorBoardLogger("lightning_logs", name="OilModel", log_graph=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from lightning.pytorch.callbacks import ModelSummary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Dataset Klasse
class PandasDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_columns: list, label_column: str):
        self.data = dataframe
        self.features = self.data[feature_columns].values
        self.labels = self.data[label_column].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Für KlassifiPandasDataModule(file_path, feature_columns, label_column)zierung
        return x, y
    



# DataModule Klasse
class PandasDataModule(L.LightningDataModule):
    def __init__(self, file_path: str, feature_columns: list, label_column: str, batch_size: int=100, train_val_split: float = 0.8):
        super().__init__()
        self.file_path = file_path
        #self.sheet_name = sheet_name
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.dataframe = pd.read_csv(self.file_path, dtype=np.float64)
        self.dataset = PandasDataset(self.dataframe, self.feature_columns, self.label_column)
        #self.generator = generator
        
    
    def prepare_data(self):
        self.dataframe = pd.read_csv(self.file_path, dtype=np.float64) #sheet_name=self.sheet_name)
    
    def setup(self, stage=None):
        self.dataset = PandasDataset(self.dataframe, self.feature_columns, self.label_column)
        train_size = int(self.train_val_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

class OilModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(OilModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        #self.fc4 = nn.Linear(hidden_size, hidden_size)
        #self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=output_size)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=output_size)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=output_size)
        self.save_hyperparameters()
        self.dropout = nn.Dropout(0.2)
        self.test_step_outputs = []
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
      
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=3e-3)
        # return optimizer
        optimizer = optim.Adam(self.parameters(), lr=3e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, 
                                                         mode='max', verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_accuracy",
                    "interval": "epoch",
                    }
                }
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        x = x.reshape(x.size(0), -1)
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        #self.log('train_loss', loss, prog_bar=True)
        train_accuracy = self.train_acc(outputs, y)
        """ self.log_dict({'train_loss': loss, 'train_accuracy': train_accuracy} ,
                      on_step=False, on_epoch=True, prog_bar=True) """
        self.log('train_accuracy', train_accuracy, on_step=False, on_epoch =True, prog_bar=True) # 'train_accuracy', train_accuracy,
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        outputs = self(x)
        loss = F.cross_entropy(outputs, y)
        #self.log('val_loss', loss, prog_bar=True)
        val_accuracy = self.valid_acc(outputs, y)
        self.log('val_accuracy',val_accuracy, on_step=False, on_epoch =True, prog_bar=True)
        return loss

    def test_step(self, batch):
        x, y = batch
        outputs = self.forward(x)
        preds = torch.argmax(outputs, dim=1)
        self.test_step_outputs.append((preds, y))
        #return {'preds': preds, 'targets': y} 

    def on_test_epoch_end(self):
        preds, targets = zip(*self.test_step_outputs)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        plot_confusion_matrix(preds.cpu().numpy(), targets.cpu().numpy())
        self.test_step_outputs.clear()    

    def predict_step(self, batch):
        x,y = batch
        y_hat = self(x)
        pred = torch.softmax(y_hat, dim =1)
        return pred       

    def on_train_epoch_end(self) -> None:
        self.print('')

    """ def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in outputs ] )
        targets = torch.cat([x['target'] for x in outputs ] )
        return preds, targets  """   






#function for plotting the confusion matrix
def plot_confusion_matrix(preds, targets):
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




def create_real_oil_samples(file_path:str):
    list_column_names = ['PaOlPa', 'PaLiPa', 'PaOlSt', 'PaOlOl', 'PaLiSt', 'PaLiOl', 'PaLiLi',
       'StOlOl', 'OlOlOl', 'StLiOl', 'OlLiOl', 'LiLiOl', 'LiLiLi', ' 14:0',
       '16:0', '16:1Δ7', '16:1Δ9', '18:0', '18:1Δ9tr', '18:1Δ9', '18:1Δ11',
       '18:2Δ9t,12t', '18:2Δ9,12', '18:3Δ9,12,15', '20:0', '20:1Δ11', '22:0',
       '22:1Δ13', '20:5Δ5,8,11,14,17', '24:0', 'α-tocopherol', 'β-tocopherol',
       'γ-tocopherol', 'β-tocotrienol', 'Plastochromanol-8', 'γ-tocotrienol',
       'δ-tocopherol']
    oil_data = get_dataframe(file_path, sheet_name='All the results')
    co, co_means, co_sd = create_subsets(oil_data, "CO")
    so, so_means, so_sd = create_subsets(oil_data, "SO")
    real_mixtures = oil_data.loc[:,list_column_names[0] :list_column_names[-2]:2].tail(34).drop(42)
    real_data = pd.concat([co_means, so_means, real_mixtures] )
    real_data['Label'] =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6,3,4,5,6,3,4,5,6,2,3,4,5,6]
    real_data.to_csv('/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/real_data_samples_mapped.csv', encoding='utf-8', index=False, header=True)
    return real_data
    
     




# Pipeline für das Training
if __name__ == "__main__":
    # Pfad zur Excel-Datei und Tabellenblattname
    file_path = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/lightning_logs/NNclassification/data_train_1000k.csv' #/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/Data_train_1k.csv
    sheet_name = 'Sheet1'
    feature_columns = ['PaOlPa', 'PaLiPa', 'PaOlSt', 'PaOlOl', 'PaLiSt', 'PaLiOl', 'PaLiLi',
       'StOlOl', 'OlOlOl', 'StLiOl', 'OlLiOl', 'LiLiOl', 'LiLiLi', ' 14:0',
       '16:0', '16:1Δ7', '16:1Δ9', '18:0', '18:1Δ9tr', '18:1Δ9', '18:1Δ11',
       '18:2Δ9t,12t', '18:2Δ9,12', '18:3Δ9,12,15', '20:0', '20:1Δ11', '22:0',
       '22:1Δ13', '20:5Δ5,8,11,14,17', '24:0', 'α-tocopherol', 'β-tocopherol',
       'γ-tocopherol', 'β-tocotrienol', 'Plastochromanol-8', 'γ-tocotrienol',
       'δ-tocopherol']  # Ersetzen mit Ihren Features
    
    label_column = 'Label'  # Ersetzen mit Ihrer Label-Spalte
    pl.seed_everything(42, workers=True)
    """ storage = 'sqlite:///Oil.db'
    
    study = optuna.create_study(direction='maximize', study_name='Oil-Study', storage=storage , load_if_exists=True)
    study.optimize(objective,  n_trials=40 ) #objective, n_trials=50, timeout=600)

    print("Best hyperparameters: ", study.best_params) """

    #Retrain Model mit besten parametern

    """ best_params = study.best_params
    hidden_layers = best_params['hidden_layers']
    hidden_dims = [best_params[f'hidden_dim_{i} '] for i in range(hidden_layers) ] """
      
    



    # DataModule erstellen
    #generator1 = torch.Generator().manual_seed(42)
    data_module = PandasDataModule(file_path, feature_columns, label_column)
    data_module.prepare_data()
    data_module.setup()
    """ data_module_test = PandasDataModule(file_path, sheet_name, feature_columns, label_column, batch_size=32)
    data_module_test.prepare_data()
    data_module_test.setup()
    data_module_test.val_dataloader """
    # Modell initialisieren
    input_size = len(feature_columns)
    hidden_size = 128
    output_size = len(data_module.dataframe[label_column].unique())  # Anzahl der Klassen
    model = OilModel(input_size, hidden_size, output_size).to(device)
    # Checkpoint Callback definieren

    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="train_accuracy", mode="max", patience=40, verbose=True)
    

    model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=1, monitor="train_accuracy", mode="max", verbose=False)




    """ checkpoint_callback = ModelCheckpoint(  
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    ) """
    # Trainer initialisieren
    trainer = L.Trainer(limit_train_batches=10, max_epochs=300, log_every_n_steps=5, logger=logger, callbacks=[early_stopping, model_checkpoint, ModelSummary(max_depth=-1)],  profiler="simple", devices=1)
    # Modell trainieren
    trainer.fit(model, data_module.train_dataloader())#, data_module.val_dataloader())
    file_path_test = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/lightning_logs/NNclassification/data_test_1000k.csv'
    test_data = PandasDataModule(file_path_test, feature_columns, label_column)
    test_data.prepare_data()
    test_data.setup()
    trainer.test(model, test_data.predict_dataloader(), ckpt_path='best')

    #trainer.test(model, data_module.val_dataloader(), ckpt_path='best')
    file_path_test_real=   '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/real_data_samples_mapped.csv'
    test_data_real = PandasDataModule(file_path_test_real, feature_columns, label_column)
    test_data_real.prepare_data()
    test_data_real.setup()

    trainer.test(model, test_data_real.predict_dataloader(), ckpt_path='best')












    #other_oils = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/other_oils_1808.csv'
    
    #predictions = trainer.predict(model, data_module.val_dataloader(), ckpt_path='best')
    #probabilities = torch.cat(predictions, dim=0)
    

    #preds, targets = model.on_test_epoch_end(test_results)
    #plot_confusion_matrix(preds, targets)

    #Predict with the Model

    #best_model = OilModel.load_from_checkpoint(model_checkpoint.best_model_path)
    #confusion_matrix(best_model, data_module)

 

    """ file_path_test=   '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/real_data_samples_mapped.csv'
    test_data_real = PandasDataModule(file_path_test, feature_columns, label_column)
    test_data_real.prepare_data()
    test_data_real.setup()
    test_trainer = L.Trainer(callbacks=[early_stopping, model_checkpoint, ModelSummary(max_depth=-1)],  profiler="simple", devices=1)
    test_trainer.test(model, test_data_real.predict_dataloader(), ckpt_path='best') """




    """ #test_data_real.prepare_data()
    #test_data_real.setup()
    best_model = OilModel.load_from_checkpoint(model_checkpoint.best_model_path, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.to(device)
    #test_loader = DataLoader(test_data_real, batch_size=32, shuffle = False)
    best_model.eval()
    trainer_test = L.Trainer()
    predictions = trainer_test.predict(best_model, dataloaders=test_data_real.predict_dataloader())
    for batch_preds in predictions:
        print(batch_preds) """



    #true_labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6,3,4,5,6,3,4,5,6,2,3,4,5,6]

    #true_labels = test_data_real[label_column].values
    #plot_confusion_matrix(predictions, true_labels) 

    """ model.eval()
    with torch.no_grad():
        pred_real = trainer.predict(model, test_loader, ckpt_path='best')
        

    print(pred_real)
    print(pred_real[0][0] )
    print(len(pred_real))
    print(pred_real[0].shape)
    print(pred_real[-1].shape)
    pred_real = torch.hstack(pred_real)  
    conf_matrix_real = confusion_matrix(true_labels, pred_real)
    report_real = classification_report(true_labels, pred_real)
    cm_real = ConfusionMatrixDisplay(conf_matrix_real, display_labels=['CO', 'SO', '99:1', '97:3', '95:5', '93:7', '91:9']).plot() """

    """ predicted_classes = inference(model, file_path_test).cpu()
    df_unseen = pd.read_csv(file_path_test)
    true_labels = df_unseen[label_column].values
    plot_confusion_matrix(predicted_classes, true_labels) """  
    
    
    
        


    """ real_samples = create_real_oil_samples(file_path_test)
    real_oil = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/real_data_samples_mapped.csv'
    prediction_real_samples = inference(model=best_model, unseen_csv_file=real_oil)
    print(prediction_real_samples, prediction_real_samples.shape, real_samples)
    plot_confusion_matrix(prediction_real_samples.cpu(), real_samples)  """   




    #prediction = best_model(data_module.val_dataloader()) 
    #other_oils = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/other_oils_1808.csv'
    #predictions_1 = inference(model=best_model, unseen_csv_file=other_oils)
    #print(predictions_1, predictions_1.shape)
     


