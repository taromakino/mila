from argparse import ArgumentParser
from copy import deepcopy
from data import get_data
from nets.wrn import *
from utils import *

class Experiment:
    def __init__(self, config):
        print(config)
        self.config = config
        set_seed(config.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data, self.val_data, self.test_data = get_data(config)
        self.net = self.get_net()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=config.lr_init, momentum=0.9, weight_decay=config.wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config.num_epochs,
            eta_min=config.lr_init * 1e-3)
        self.optimal_val_acc = -np.Inf
        self.optimal_weights = deepcopy(self.net.state_dict())
        if os.path.exists(self.config.results_path):
            self.load_checkpoint()
        else:
            os.makedirs(self.config.results_path)
            save_file(config, os.path.join(self.config.results_path, 'config.pkl'))
            self.epoch = 0

    def get_net(self):
        if self.config.arch =='wrn':
            return WideResNet(40, 2, 10, 1).to(self.device)
        else:
            raise NotImplementedError

    def save_checkpoint(self):
        checkpoint = {
            'random_state_np': np.random.get_state(),
            'random_state_pt': torch.get_rng_state(),
            'random_state': random.getstate(),
            'net_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': self.epoch}
        save_file(checkpoint, os.path.join(self.config.results_path, 'checkpoint.pkl'))

    def load_checkpoint(self):
        print('Loading checkpoint')
        checkpoint = load_file(os.path.join(self.config.results_path, 'checkpoint.pkl'))
        np.random.set_state(checkpoint['random_state_np'])
        torch.set_rng_state(checkpoint['random_state_pt'])
        random.setstate(checkpoint['random_state'])
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']

    def train_epoch(self):
        train_loss = train_acc = 0
        self.net.train()
        for x, y in self.train_data:
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

    def to_summary_str(self, loss, acc):
        return '{}, {}, {:.6f}, {:.1f}%'.format(
            get_time(),
            self.epoch,
            loss,
            100 * acc)

    def train(self):
        for epoch in range(self.epoch, self.config.num_epochs):
            train_loss, train_acc = self.train_epoch()
            self.scheduler.step(epoch)
            write(os.path.join(self.config.results_path, 'train_summary.txt'), self.to_summary_str(train_loss, train_acc))
            self.epoch += 1
            if self.epoch > 0 and self.epoch % self.config.val_interval == 0:
                # Evaluate on validation set and store optimal weights
                val_loss, val_acc = self.inference(True)
                write(os.path.join(self.config.results_path, 'val_summary.txt'), self.to_summary_str(val_loss, val_acc))
                if val_acc > self.optimal_val_acc:
                    self.optimal_weights = deepcopy(self.net.state_dict())
            if self.epoch > 0 and self.epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        # At the end of training, load optimal weights and evaluate on test set
        self.net.load_state_dict(self.optimal_weights)
        test_loss, test_acc = self.inference(False)
        write(os.path.join(self.config.results_path, 'test_summary.txt'), self.to_summary_str(test_loss, test_acc))

    def inference(self, is_val):
        inference_data = self.val_data if is_val else self.test_data
        loss = acc = 0
        self.net.eval()
        with torch.no_grad():
            for x, y in inference_data:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.net(x)
                loss += F.cross_entropy(logits, y, reduction='sum').item()
                acc += (logits.argmax(1) == y).sum().item()
        loss /= len(inference_data.dataset)
        acc /= len(inference_data.dataset)
        return loss, acc

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--arch', default='wrn')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr_init', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    exp = Experiment(get_config())
    exp.train()