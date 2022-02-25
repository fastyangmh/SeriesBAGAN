#import
from argparse import Namespace
from os.path import join
from glob import glob
import pandas as pd
from FlowCal.io import FCSData
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import TypeVar
from pytorch_lightning import LightningDataModule
from typing import Optional, Union, List, Dict
import torch
import random

T_co = TypeVar('T_co', covariant=True)


#def
def create_datamodule(project_parameters):
    return MyLightningDataModule(root=project_parameters.root,
                                 classes=project_parameters.classes,
                                 val_size=project_parameters.val_size,
                                 batch_size=project_parameters.batch_size,
                                 num_workers=project_parameters.num_workers,
                                 device=project_parameters.device,
                                 max_samples=project_parameters.max_samples)


#class
class MyDataset(Dataset):
    def __init__(self, data, label) -> None:
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        # the data type of sample is float32
        sample = self.data[index]
        #convert the value of sample to 0~1
        #NOTE: np.finfo(dtype=np.float32).max = 3.4028235e+38
        #sample = sample / np.finfo(dtype=np.float32).max
        sample = (sample - sample.min()) / (sample.max() - sample.min())
        target = self.label[index]
        #the shape of sample is (31,)
        #the shape of target is ()
        return sample, target


class MyLightningDataModule(LightningDataModule):
    def __init__(self, root, classes, val_size, batch_size, num_workers,
                 device, max_samples):
        super().__init__()
        self.root = root
        self.classes = classes
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = device == 'cuda' and torch.cuda.is_available()
        self.max_samples = max_samples

    def prepare_data(self) -> None:
        data = []
        label = []
        files = sorted(glob(join(self.root, 'raw_fcs/*/*.fcs')))
        df = pd.read_excel(join(self.root, 'EU_label.xlsx'))
        use_indices = pd.read_excel(join(self.root,
                                         'EU_marker_channel_mapping.xlsx'),
                                    usecols=['use'])
        use_indices = np.where(use_indices.values == 1)[0].tolist()
        for f in files:
            data.append(np.array(FCSData(infile=f)[:, use_indices]))
            l = df.loc[df.file_flow_id == f.split('/')[3], 'label'].item()
            l = np.zeros(shape=len(data[-1]),
                         dtype=np.int16) + self.class_to_idx[l]
            label.append(l)
        data = np.concatenate(data)
        label = np.concatenate(label)
        if self.max_samples is not None:
            index = random.sample(population=range(len(data)),
                                  k=self.max_samples)
            data = data[index]
            label = label[index]
        self.data = data
        self.label = label

    def setup(self, stage: Optional[str] = None) -> None:
        x_train, x_val, y_train, y_val = train_test_split(
            self.data, self.label, test_size=self.val_size)
        self.train_dataset = MyDataset(data=x_train, label=y_train)
        self.val_dataset = MyDataset(data=x_val, label=y_val)

    def train_dataloader(
            self
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


#class

if __name__ == '__main__':
    #project_parameters
    project_parameters = Namespace(
        **{
            'root': 'data/FCS_data/',
            'classes': ['Healthy', 'Sick'],
            'batch_size': 32,
            'val_size': 0.2,
            'num_workers': 0,
            'device': 'cpu',
            'max_samples': None
        })

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print('the dimension of target: {}'.format(y.shape))
