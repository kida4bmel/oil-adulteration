import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchmetrics
from utility_functions import plot_confusion_matrix_and_report

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
        plot_confusion_matrix_and_report(preds.cpu().numpy(), targets.cpu().numpy())
        self.test_step_outputs.clear()    
    def predict_step(self, batch):
        x,y = batch
        y_hat = self(x)
        pred = torch.softmax(y_hat, dim =1)
        return pred       
    def on_train_epoch_end(self) -> None:
        self.print('')
