import torch
import time

class Trainer():
  def __init__(self, model, device, is_logkey, criterion, lr, optim, train_data_loader, valid_data_loader):
    self.model = model
    self.device = device
    self.is_logkey = is_logkey
    self.criterion = criterion
    self.train_loss = []
    self.valid_loss = []
    self.learning_rate = lr
    self.optimizer = optim
    self.train_data_loader = train_data_loader
    self.valid_data_loader = valid_data_loader

  def iteration(self, epoch, start_train):
    total_dist = []
    total_loss = 0.0

    data_loader = self.train_data_loader if start_train else self.valid_data_loader
    total_length = len(data_loader)
    data_iter = enumerate(data_loader)

    start = time.strftime("%H:%M:%S")
    start_time = time.time()

    for i, data in data_iter:
        data = {key: value.to(self.device) for key, value in data.items()}

        result = self.model.forward(data["bert_input"], data["time_input"])
        mask_lm_output, mask_time_output = result["logkey_output"], result["time_output"]

        # NLLLoss of predicting masked token word ignore_index = 0 to ignore unmasked tokens
        loss = torch.tensor(0) if not self.is_logkey else self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

        total_loss += loss.item()

        # backward and optimization for training run only
        if (start_train):
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

    avg_loss = total_loss / total_length
    epoch_time = time.time() - start_time
    if start_train:
        self.train_loss.append({"epoch": epoch, "lr": self.learning_rate, "time": start, "loss": avg_loss})
        print(f'Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec')
    else:
        self.valid_loss.append({"epoch": epoch, "lr": self.learning_rate, "time": start, "loss": avg_loss})
        print(f'Epoch: {epoch + 1}, Valid Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} sec')

    return avg_loss, total_dist

  def train(self, epoch):
    return self.iteration(epoch, start_train=True)

  def valid(self, epoch):
    return self.iteration(epoch, start_train=False)