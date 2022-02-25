#import
from argparse import Namespace
from pytorch_lightning import seed_everything
from data_preparation import create_datamodule
from pytorch_lightning import LightningModule
from model import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import torch
from VAE import create_trainer
import matplotlib.pyplot as plt


#def
def create_gan_model(project_parameters):
    return GAN(in_features=project_parameters.in_features,
               z_dim=project_parameters.z_dim,
               classes=project_parameters.classes,
               lr_g=project_parameters.lr_g,
               lr_d=project_parameters.lr_d,
               T_max_g=project_parameters.T_max_g,
               T_max_d=project_parameters.T_max_d,
               checkpoint_path=project_parameters.checkpoint_path)


#class
class GAN(LightningModule):
    def __init__(self, in_features, z_dim, classes, lr_g, lr_d, T_max_g,
                 T_max_d, checkpoint_path) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.generator = Generator(in_features=in_features, z_dim=z_dim)
        self.generator = self.load_weight(model=self.generator, typ='decoder')
        self.discriminator = Discriminator(in_features=in_features,
                                           z_dim=z_dim,
                                           num_classes=len(classes))
        self.discriminator = self.load_weight(model=self.discriminator,
                                              typ='encoder')
        self.loss_function = nn.BCEWithLogitsLoss()
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.T_max_g = T_max_g
        self.T_max_d = T_max_d
        self.distribution = torch.load('distribution.pt')
        self.classes = classes

    def load_weight(self, model, typ):
        checkpoint = torch.load(self.checkpoint_path)
        weight = checkpoint['state_dict']
        weight_keys = list(weight.keys())
        for k in weight_keys:
            if typ not in k:
                weight.pop(k)
        weight = {
            k: v
            for k, v in zip(model.state_dict().keys(), weight.values())
        }
        model.load_state_dict(state_dict=weight, strict=False)
        return model

    def configure_optimizers(self):
        optimizer_g = optim.Adam(params=self.generator.parameters(),
                                 lr=self.lr_g)
        optimizer_d = optim.Adam(params=self.discriminator.parameters(),
                                 lr=self.lr_d)
        lr_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer_g, T_max=self.T_max_g)
        lr_scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer_d, T_max=self.T_max_d)
        return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]

    def forward(self, c, idx=None):
        if idx is None:
            idx = torch.randint(low=0,
                                high=len(self.distribution[c]),
                                size=(1, ))
        z = self.distribution[c][idx]
        return self.generator(z)

    def shared_step(self, batch):
        x, y = batch
        z = []
        for v in y:
            v = v.item()
            idx = torch.randint(low=0,
                                high=len(self.distribution[v]),
                                size=(1, )).item()
            z.append(self.distribution[v][idx][None])
        z = torch.cat(z, 0)
        y = torch.eye(len(self.classes))[y.long()]
        x_hat = self.generator(z)
        return x, x_hat, y, z

    def training_step(self, batch, batch_idx, optimizer_idx):
        #update d
        x, x_hat, y, z = self.shared_step(batch=batch)
        if optimizer_idx == 1:
            _, y_hat_real = self.discriminator(x)
            _, y_hat_fake = self.discriminator(x_hat.detach())
            y_real = torch.cat([torch.ones((len(y), 1)), y], -1)
            y_fake = torch.cat([torch.zeros((len(y), 1)), y], -1)
            real_loss = self.loss_function(y_hat_real, y_real)
            fake_loss = self.loss_function(y_hat_fake, y_fake)
            d_loss = real_loss + fake_loss
            self.log('train_d_loss',
                     d_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return d_loss
        #update g
        if optimizer_idx == 0:
            _, y_hat_fake = self.discriminator(x_hat)
            y_real = torch.cat([torch.ones((len(y), 1)), y], -1)
            g_loss = self.loss_function(y_hat_fake, y_real)
            self.log('train_g_loss',
                     g_loss,
                     on_step=True,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
            return g_loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, y, z = self.shared_step(batch=batch)
        _, y_hat_fake = self.discriminator(x_hat)
        y_fake = torch.cat([torch.zeros((len(y), 1)), y], -1)
        g_loss = self.loss_function(y_hat_fake, y_fake)
        self.log('val_loss', g_loss)
        return g_loss

    def test_step(self, batch, batch_idx):
        x, x_hat, y, z = self.shared_step(batch=batch)
        _, y_hat_fake = self.discriminator(x_hat)
        y_fake = torch.cat([torch.zeros((len(y), 1)), y], -1)
        g_loss = self.loss_function(y_hat_fake, y_fake)
        self.log('test_loss', g_loss)
        return g_loss


if __name__ == '__main__':
    #project parameters
    project_parameters = Namespace(
        **{
            'root': 'data/FCS_data/',
            'classes': ['Healthy', 'Sick'],
            'batch_size': 32,
            'val_size': 0.2,
            'num_workers': 0,
            'device': 'cpu',
            'max_samples': 1000,
            'random_seed': 0,
            'in_features': 31,
            'z_dim': 10,
            'lr_g': 1e-3,
            'lr_d': 1e-3,
            'T_max_g': 10,
            'T_max_d': 10,
            'patience': 3,
            'early_stopping': True,
            'max_epochs': 100,
            'random_seed': 0,
            'checkpoint_path': 'VAE_pretrained.ckpt'
        })

    #set random seed
    seed_everything(seed=project_parameters.random_seed)

    #create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    #create model
    model = create_gan_model(project_parameters=project_parameters)

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

    #generate
    model = model.eval()
    with torch.no_grad():
        x, y = next(iter(datamodule.val_dataloader()))
        x_hat = model(y[0].item())
        x = x[0][None]

    #display
    plt.plot(x[0], label='x')
    plt.plot(x_hat[0], label='x_hat')
    plt.legend()
    plt.show()
