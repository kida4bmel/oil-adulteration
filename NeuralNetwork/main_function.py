import torch
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
torch.set_float32_matmul_precision('high')
from lightning.pytorch.loggers import TensorBoardLogger 
logger = TensorBoardLogger("lightning_logs", name="OilModel", log_graph=True)
torch.multiprocessing.set_sharing_strategy('file_system')
from lightning.pytorch.callbacks import ModelSummary
from data_modules import *
from neural_network import *
from utility_functions import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pytorch_lightning as pl


if __name__ == "__main__":
    # Pfad zur Excel-Datei und Tabellenblattname
    pl.seed_everything(42, workers=True)
    file_path = ''
    sheet_name ='yourSheetName' #Sheet1
    feature_columns = ['PaOlPa', 'PaLiPa', 'PaOlSt', 'PaOlOl', 'PaLiSt', 'PaLiOl', 'PaLiLi',
       'StOlOl', 'OlOlOl', 'StLiOl', 'OlLiOl', 'LiLiOl', 'LiLiLi', ' 14:0',
       '16:0', '16:1Δ7', '16:1Δ9', '18:0', '18:1Δ9tr', '18:1Δ9', '18:1Δ11',
       '18:2Δ9t,12t', '18:2Δ9,12', '18:3Δ9,12,15', '20:0', '20:1Δ11', '22:0',
       '22:1Δ13', '20:5Δ5,8,11,14,17', '24:0', 'α-tocopherol', 'β-tocopherol',
       'γ-tocopherol', 'β-tocotrienol', 'Plastochromanol-8', 'γ-tocotrienol',
       'δ-tocopherol']  # Ersetzen mit Ihren Features  
    label_column = 'Label'  # Ersetzen mit Ihrer Label-Spalte
    # DataModule erstellen
    data_module = PandasDataModule(file_path, feature_columns, label_column)
    data_module.prepare_data()
    data_module.setup()
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
    # Trainer initialisieren
    trainer = L.Trainer(limit_train_batches=10, max_epochs=300, log_every_n_steps=5, logger=logger, callbacks=[early_stopping, model_checkpoint, ModelSummary(max_depth=-1)],  profiler="simple", devices=1)
    # Modell trainieren
    trainer.fit(model, data_module.train_dataloader())#, data_module.val_dataloader())
    file_path_test = '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/lightning_logs/NNclassification/data_test_1k.csv'
    test_data = PandasDataModule(file_path_test, feature_columns, label_column)
    test_data.prepare_data()
    test_data.setup()
    trainer.test(model, test_data.predict_dataloader(), ckpt_path='best')
    file_path_test_real=   '/home/roggenlanda/Schreibtisch/ServiceRequests/Oil/nnClassification/real_data_samples_mapped.csv'
    test_data_real = PandasDataModule(file_path_test_real, feature_columns, label_column)
    test_data_real.prepare_data()
    test_data_real.setup()
    trainer.test(model, test_data_real.predict_dataloader(), ckpt_path='best')