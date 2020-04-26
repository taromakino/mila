import gin
import sys

from copy import deepcopy
from data import get_data
from nets.wrn import *
from torch.optim import Adam, SGD
from torchvision.datasets import CIFAR10, CIFAR100
from utils import *

gin.external_configurable(CIFAR10)
gin.external_configurable(CIFAR100)
gin.external_configurable(WideResNet)
gin.external_configurable(Adam)
gin.external_configurable(SGD)

@gin.configurable
class Experiment:
    '''
    Train for a specified number of epochs, while intermittently saving checkpoints and keeping track of the optimal
    weights with respect to validation performance. At the end of training, load the optimal weights and perform inference
    on the test set.
    '''
    def __init__(self,
                 save_dpath,
                 seed,
                 stages,
                 dataset,
                 arch_class,
                 arch_kwargs,
                 optimizer_class,
                 optimizer_kwargs,
                 num_epochs,
                 batch_size,
                 initial_weights_fpath,
                 resuming):
        self.save_dpath = save_dpath
        set_seed(seed)
        self.stages = stages
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data, self.val_data, self.test_data = get_data(dataset, batch_size)
        self.net = arch_class(**arch_kwargs)
        if initial_weights_fpath is not None:
            self.net.load_state_dict(torch.load(initial_weights_fpath))
        self.net.to(self.device)
        self.optimizer = optimizer_class(self.net.parameters(), **optimizer_kwargs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs, eta_min=optimizer_kwargs['lr'] * 1e-3)
        self.optimal_val_acc = -np.Inf
        self.optimal_weights = deepcopy(self.net.state_dict())
        if resuming:
            self.load_checkpoint()
        else:
            os.makedirs(save_dpath)
            self.epoch = 0

    def save_checkpoint(self):
        '''
        Save the current state.
        '''
        checkpoint = {
            'random_state_np': np.random.get_state(),
            'random_state_pt': torch.get_rng_state(),
            'random_state': random.getstate(),
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': self.epoch}
        save_file(checkpoint, os.path.join(self.save_dpath, 'checkpoint.pkl'))

    def load_checkpoint(self):
        '''
        Load a previously saved state.
        '''
        print('Loading checkpoint')
        checkpoint = load_file(os.path.join(self.save_dpath, 'checkpoint.pkl'))
        np.random.set_state(checkpoint['random_state_np'])
        torch.set_rng_state(checkpoint['random_state_pt'])
        random.setstate(checkpoint['random_state'])
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']

    def to_summary_str(self, loss, acc):
        '''
        Return the loss and accuracy in a readable format.
        '''
        return '{}, {}, {:.6f}, {:.1f}%'.format(
            get_time(),
            self.epoch,
            loss,
            100 * acc)

    def train_epoch(self):
        '''
        Train for a single epoch.
        '''
        train_loss = train_acc = 0
        self.net.train()
        for x, y, idxs in self.train_data:
            # Forward prop
            x, y = x.to(self.device), y.to(self.device)
            logits = self.net(x)
            loss = F.cross_entropy(logits, y, reduction='none')
            # Backprop
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # Summary
            train_loss += loss.sum().item()
            train_acc += (logits.argmax(1) == y).sum().item()
        return train_loss / len(self.train_data.dataset), train_acc / len(self.train_data.dataset)

    def inference(self, is_val):
        '''
        Perform inference on the validation or test set.
        '''
        inference_data = self.val_data if is_val else self.test_data
        loss = acc = 0
        logits_all = np.full((len(inference_data.dataset), inference_data.dataset.y.max().item() + 1), np.nan)
        y_all = np.full(len(inference_data.dataset), np.nan)
        self.net.eval()
        with torch.no_grad():
            for x, y, idxs in inference_data:
                x, y = x.to(self.device), y.to(self.device)
                y_all[idxs] = y.cpu().numpy()
                logits = self.net(x)
                logits_all[idxs] = logits.cpu().numpy()
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                acc += (logits.argmax(1) == y).sum().item()
        save_file(logits_all, os.path.join(self.save_dpath, f"logits_{'val' if is_val else 'test'}.pkl"))
        save_file(y_all, os.path.join(self.save_dpath, f"y_{'val' if is_val else 'test'}.pkl"))
        loss /= len(inference_data.dataset)
        acc /= len(inference_data.dataset)
        return loss, acc

    def run(self):
        '''
        See the constructor docstring.
        '''
        for epoch in range(self.epoch, self.num_epochs):
            if 'train' in self.stages:
                train_loss, train_acc = self.train_epoch()
                self.scheduler.step(epoch)
                write(os.path.join(self.save_dpath, 'train_summary.txt'), self.to_summary_str(train_loss, train_acc))
                self.save_checkpoint()
            if 'val' in self.stages:
                val_loss, val_acc = self.inference(True)
                write(os.path.join(self.save_dpath, 'val_summary.txt'), self.to_summary_str(val_loss, val_acc))
                if val_acc > self.optimal_val_acc:
                    self.optimal_weights = deepcopy(self.net.state_dict())
            self.epoch += 1
        if 'train' in self.stages:
            self.save_checkpoint()
            torch.save(self.optimal_weights, os.path.join(self.save_dpath, 'optimal_weights.pkl'))
        if 'test' in self.stages:
            self.net.load_state_dict(self.optimal_weights)
            test_loss, test_acc = self.inference(False)
            write(os.path.join(self.save_dpath, 'test_summary.txt'), self.to_summary_str(test_loss, test_acc))

if __name__ == '__main__':
    config_path = sys.argv[-1]
    gin.parse_config_file(config_path)
    Experiment().run()