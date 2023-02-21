import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

class Cifar10DataLoader(Dataset):
    """
    loading cifar10 dataset and return dataloader

    """
    def __init__(self, 
                train:bool,
                data_path:str = '~/data',
                transform:transforms.Compose = None
                ) -> None:
        """initializing cifar 10 dataset

        Args:
            train (bool): train flag for cifar10 dataset
            data_path (str, optional): . Defaults to '~/data'.
            transform (transforms.Compose, optional): _description_. Defaults to None.
        """
        super(Cifar10DataLoader, self).__init__()
        
        if transform:
            self.cifar_data = datasets.CIFAR10(root=data_path, train=train, download=True, transform=transform)
        else :
            self.cifar_data = datasets.CIFAR10(root=data_path, train=train, download=True)
    
    def _split_validation(self, split_ratio):
        """split data to train and valid dataset using split_ratio

        Args:
            split_ratio (float): Default to 0.8.
        """
        train_length = int(len(self.cifar_data) * split_ratio)
        self.train_data, self.valid_data = random_split(
            self.cifar_data,
            [train_length, len(self.cifar_data)-train_length]
        )
    
    def get_dataloader(self, 
                        batch_size:int,
                        num_workers:int,
                        shuffle:bool=True,
                        split_validation:bool=True,
                        split_ratio:float=0.8
                        ) -> DataLoader:
        """make dataloader and return it

        Args:
            batch_size (int): batch_size for dataloader
            num_workers (int): num_workers for dataloader
            shuffle (bool, optional): shuffle flag. Defaults to True.
            split_validation (bool, optional): flag for split dataset to train and validation. Defaults to True.
            split_ratio (float, optional): Defaults to 0.8.

        Returns:
            DataLoader: if using split_validation, it will return train, valid dataloader
        """
        if split_validation:
            self._split_validation(split_ratio)
            
            train_dataloader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=shuffle, 
                num_workers=num_workers, pin_memory=True)
            valid_dataloader = DataLoader(
                self.valid_data, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, pin_memory=True
            )
            return train_dataloader, valid_dataloader
        else:
            cifar_dataloader = DataLoader(
                self.cifar_data, batch_sampler=batch_size,shuffle=shuffle,
                num_workers=num_workers, pin_memory=True
            )
            return cifar_dataloader