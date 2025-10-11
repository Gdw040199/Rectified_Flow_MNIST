import torch
from model import MiniUNet
from rectified_flow import RectifiedFlow
import cv2
import os
import numpy as np


def infer(
        checkpoint_path,
        base_channels=16,
        step=60,  
        num_imgs=5,
        y=None,
        cfg_scale=7.0,
        save_path='./results',
        save_noise_path=None,
        device='cuda'):
    """
    Use the trained model to generate images.
    Args:
        checkpoint_path: Path to the trained model checkpoint.
        base_channels: Base number of channels in the MiniUNet model.
        step: Number of steps for the Euler method.
        num_imgs: Number of images to generate.
        y: Conditional labels for image generation. If None, generate unconditionally.
        cfg_scale: Classifier-free guidance scale. Larger values lead to stronger conditioning.
        save_path: Directory to save the generated images.
        save_noise_path: Directory to save the initial noise. If None, do not save.
        device: Device to run the model on, e.g., 'cuda' or 'cpu

    """
    os.makedirs(save_path, exist_ok=True)
    if save_noise_path is not None:
        os.makedirs(save_noise_path, exist_ok=True)

    if y is not None:
        assert len(y.shape) == 1 or len(
            y.shape) == 2, 'y must be 1D or 2D tensor'
        assert y.shape[0] == num_imgs or y.shape[
            0] == 1, 'y.shape[0] must be equal to num_imgs or 1'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        y = y.to(device)

    model = MiniUNet(base_channels=base_channels)
    model.to(device)
    model.eval()

    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])


    with torch.no_grad():
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            dt = 1.0 / step
            # generate initial noise
            x_t = torch.randn(1, 1, 28, 28).to(device)
            noise = x_t.detach().cpu().numpy()

            # get the conditional label for this image
            if y is not None:
                y_i = y[i].unsqueeze(0)

            for j in range(step):
                if j % 10 == 0:
                    print(f'Generating {i}th image, step {j}...')
                t = j * dt
                t = torch.tensor([t]).to(device)

                if y is not None:
                    # x = x_uncond + cfg_scale * (x_cond - x_uncond)
                    v_pred_uncond = model(x=x_t, t=t)
                    v_pred_cond = model(x=x_t, t=t, y=y_i)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond) # cfg
                else:
                    v_pred = model(x=x_t, t=t)

                # Euler method to update x_t
                x_t = rf.euler(x_t, v_pred, dt)

            x_t = x_t[0]
            # normalize to [0, 1]
            x_t = (x_t / 2 + 0.5).clamp(0, 1)
            # x_t = x_t.clamp(0, 1)
            img = x_t.detach().cpu().numpy()
            img = img[0] * 255
            img = img.astype('uint8')
            cv2.imwrite(os.path.join(save_path, f'{i}.png'), img)
            if save_noise_path is not None:
                np.save(os.path.join(save_noise_path, f'{i}.npy'), noise)


if __name__ == '__main__':
    y = []
    for i in range(10):
        y.extend([i] * 10)
    # v1.1 1-RF
    infer(checkpoint_path='./checkpoints/v1.1-cfg/miniunet_1.pth',
          base_channels=64,
          step=2,
          num_imgs=100,
          y=torch.tensor(y),
          cfg_scale=5.0,
          save_path='./results/cfg',
          device='cuda')

    """
    # v1.2 2-RF
    infer(checkpoint_path='./checkpoints/v1.2-reflow-cfg/miniunet_19.pth',
          base_channels=64,
          step=2,
          num_imgs=100,
          y=torch.tensor(y),
          cfg_scale=5.0,
          save_path='./results/reflow-cfg',
          device='cuda')
    """
