import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from utils import *

class PILDataset(Dataset):
    '''
    x is PIL, y is tensor
    '''
    def __init__(self, x, y, transformations):
        self.x = x
        self.y = y
        self.transformations = transformations

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = self.transformations(x)
        return x, y

    def __len__(self):
        return len(self.x)

def save_cifar10(train_ratio=0.8):
    '''
    Since CIFAR-10 doesn't come with a validation set, create one here and save the arrays to disk.
    '''
    print('Saving cifar10')
    trainval_dataset = CIFAR10(os.environ['DATA_PATH'], train=True)
    x_trainval, y_trainval = [], []
    for x, y in trainval_dataset:
        x_trainval.append(x)
        y_trainval.append(y)
    x_trainval, y_trainval = np.stack(x_trainval), np.stack(y_trainval)
    np.random.seed(0) # Always use the same validation set
    train_idxs = np.random.choice(len(x_trainval), int(train_ratio * len(x_trainval)), replace=False)
    val_idxs = np.setdiff1d(np.arange(len(x_trainval)), train_idxs)
    x_train, y_train = x_trainval[train_idxs], y_trainval[train_idxs]
    x_val, y_val = x_trainval[val_idxs], y_trainval[val_idxs]
    test_dataset = CIFAR10(os.environ['DATA_PATH'], train=False)
    x_test, y_test = [], []
    for x, y in test_dataset:
        x_test.append(x)
        y_test.append(y)
    x_test, y_test = np.stack(x_test), np.stack(y_test)
    save_file((x_train, y_train, x_val, y_val, x_test, y_test), os.path.join(os.environ['DATA_PATH'], 'cifar10.pkl'))

def get_cifar10(batch_size):
    '''
    Get dataloaders for train, validation, and test
    '''
    fpath = os.path.join(os.environ['DATA_PATH'], 'cifar10.pkl')
    if not os.path.exists(fpath):
        save_cifar10()
    x_train, y_train, x_val, y_val, x_test, y_test = load_file(fpath)
    # Train
    x_train = [Image.fromarray(elem) for elem in x_train]
    y_train = torch.tensor(y_train, dtype=torch.long)
    transformations = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    train_dataset = PILDataset(x_train, y_train, transformations)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Validation
    x_val = [Image.fromarray(elem) for elem in x_val]
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_dataset = PILDataset(x_val, y_val, transforms.ToTensor())
    val_data = DataLoader(val_dataset, batch_size=batch_size)
    # Test
    x_test = [Image.fromarray(elem) for elem in x_test]
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = PILDataset(x_test, y_test, transforms.ToTensor())
    test_data = DataLoader(test_dataset, batch_size=batch_size)
    return train_data, val_data, test_data

def get_data(dataset, batch_size):
    if dataset == 'cifar10':
        return get_cifar10(batch_size)
    else:
        raise NotImplementedError