#import
from argparse import Namespace
from pytorch_lightning import LightningModule, Trainer, seed_everything
from model import Encoder, Decoder
import torch.nn as nn
import torch.optim as optim
import torch
from data_preparation import create_datamodule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from shutil import copy2


#def
def create_vae_model(project_parameters):
    return VAE(in_features=project_parameters.in_features,
               z_dim=project_parameters.z_dim,
               lr=project_parameters.lr,
               T_max=project_parameters.T_max)


def create_trainer(project_parameters):
    if project_parameters.device == 'cuda' and torch.cuda.is_available():
        accelerator = 'gpu'
        gpus = 1
    else:
        accelerator = 'cpu'
        gpus = 0
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.4f}',
                        monitor='val_loss',
                        mode='min')
    ]
    if project_parameters.early_stopping:
        callbacks.append(
            EarlyStopping(monitor='val_loss',
                          patience=project_parameters.patience,
                          mode='min'))
    return Trainer(accelerator=accelerator,
                   callbacks=callbacks,
                   check_val_every_n_epoch=1,
                   default_root_dir='save/',
                   deterministic=True,
                   gpus=gpus,
                   max_epochs=project_parameters.max_epochs)


#class
class VAE(LightningModule):
    def __init__(self, in_features, z_dim, lr, T_max) -> None:
        super().__init__()
        self.encoder = Encoder(in_features=in_features, z_dim=z_dim)
        self.decoder = Decoder(in_features=in_features, z_dim=z_dim)
        self.mse_loss = nn.MSELoss()
        self.lr = lr
        self.T_max = T_max

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.T_max)
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        mu, sigma = self.encoder(x)
        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(z)
        return x_hat

    def shared_step(self, batch):
        x, _ = batch
        mu, sigma = self.encoder(x)
        std = torch.exp(sigma / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decoder(z)
        kl_loss = torch.mean(-0.5 *
                             torch.sum(1 + sigma - mu**2 - sigma.exp(), dim=1),
                             dim=0)
        rec_loss = self.mse_loss(x_hat, x)
        loss = kl_loss + rec_loss
        return x, x_hat, loss

    def training_step(self, batch, batch_idx):
        x, x_hat, loss = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, loss = self.shared_step(batch=batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, x_hat, loss = self.shared_step(batch=batch)
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':
    #project_parameters
    project_parameters = Namespace(
        **{
            'root': 'data/FCS_data/',
            'classes': ['Healthy', 'Sick'],
            'batch_size': 32,
            'val_size': 0.2,
            'num_workers': 0,
            'device': 'cuda',
            'max_samples': 1000,
            'in_features': 31,
            'z_dim': 10,
            'lr': 1e-2,
            'T_max': 10,
            'patience': 3,
            'early_stopping': True,
            'max_epochs': 100,
            'random_seed': 0
        })

    # set random seed
    seed_everything(seed=project_parameters.random_seed)

    #create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    #create model
    model = create_vae_model(project_parameters=project_parameters)

    #create trainer
    trainer = create_trainer(project_parameters=project_parameters)

    #training
    trainer.fit(model=model, datamodule=datamodule)

    #evalute
    dataloaders_dict = {
        'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
        'test': datamodule.test_dataloader()
    }
    result = {'trainer': trainer, 'model': model}
    for stage, dataloader in dataloaders_dict.items():
        result[stage] = trainer.test(dataloaders=dataloader,
                                     ckpt_path='best')[0]

    #copy pretrained model
    copy2(src=trainer.callbacks[-1].best_model_path,
          dst='./VAE_pretrained.ckpt')

    #get first batch in validation dataset
    model = model.eval()
    with torch.no_grad():
        x, y = next(iter(datamodule.val_dataloader()))
        x_hat = model(x).cpu().data.numpy()
        z = model.encoder(x)

    #display
    plt.plot(x[0], label='x')
    plt.plot(x_hat[0], label='x_hat')
    plt.legend()
    plt.show()

    #get distribution of training dataset
    model = model.eval()
    distribution = []
    y_true = []
    with torch.no_grad():
        for x, y in datamodule.train_dataloader():
            mu, sigma = model.encoder(x)
            std = torch.exp(sigma / 2)
            eps = torch.randn_like(std)
            z = mu + eps * std
            distribution.append(z)
            y_true.append(y)
    distribution = torch.cat(distribution, 0)
    y_true = torch.cat(y_true, 0)
    temp = {}
    for idx, c in enumerate(project_parameters.classes):
        temp[idx] = distribution[y_true == idx]
    distribution = temp
    torch.save(distribution, 'distribution.pt')
