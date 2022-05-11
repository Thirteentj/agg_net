from utils import load_data
from model import MLPClassifier, save_model
from os import path
import torch
from torch.utils.tensorboard import SummaryWriter

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MLPClassifier().to(device)
    epochs = 250
    train_logger = SummaryWriter(path.join('./log', 'train'), flush_secs=1)
    """
    Your code here

    """
    data = load_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_function = torch.nn.MSELoss()
    global_step = 0
    for epoch in range(epochs):
        print("Epoch #: ", epoch)
        for state, label in data:
            state = state.type(torch.FloatTensor).to(device)
            label = label.type(torch.FloatTensor).to(device)
            logit = model(state).reshape((-1, 2, 5))
            
            loss = loss_function(logit, label)
            train_logger.add_scalar('loss', loss, global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
    save_model(model)


if __name__ == '__main__':
    train()
