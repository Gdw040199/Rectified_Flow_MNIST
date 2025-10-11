import torch
import os
import yaml
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from model import MiniUNet
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from rectified_flow import RectifiedFlow


def train(config: str):
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    base_channels = config.get('base_channels', 16)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print_interval = config.get('batch_print_interval', 100)
    checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    use_cfg = config.get('use_cfg', False)
    device = config.get('device', 'cuda')

    # print config
    print('Training config:')
    print(f'base_channels: {base_channels}')
    print(f'epochs: {epochs}')
    print(f'batch_size: {batch_size}')
    print(f'lr_adjust_epoch: {lr_adjust_epoch}')
    print(f'batch_print_interval: {batch_print_interval}')
    print(f'checkpoint_save_interval: {checkpoint_save_interval}')
    print(f'save_path: {save_path}')
    print(f'use_cfg: {use_cfg}')
    print(f'device: {device}')

    transform = Compose([ToTensor()]) # , Normalize((0.5,), (0.5,))])

    dataset = MNIST(
        root='./data',
        train=True,  
        download=True,
        transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    model = MiniUNet(base_channels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)
    rf = RectifiedFlow()
    loss_list = []
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):

            x_1, y = data  
            t = torch.rand(x_1.size(0))
            x_t, x_0 = rf.create_flow(x_1, t)
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)
            t = t.to(device)
            optimizer.zero_grad()

            if use_cfg:
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                y = torch.cat([y, -torch.ones_like(y)], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                y = y.to(device)
            else:
                y = None

            v_pred = model(x=x_t, t=t, y=y)

            loss = rf.mse_loss(v_pred, x_1, x_0)

            loss.backward()
            optimizer.step()

            if batch % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {batch}] loss: {loss.item()}')

            loss_list.append(loss.item())

        scheduler.step()

        if epoch % checkpoint_save_interval == 0 or epoch == epochs - 1 or epoch == 0:
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'miniunet_{epoch + 1}.pth'))


if __name__ == '__main__':
    train(config='./config/train_mnist.yaml')
