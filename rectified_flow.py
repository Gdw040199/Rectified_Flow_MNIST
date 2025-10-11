import torch
import torch.nn as nn
import torch.nn.functional as F

class RectifiedFlow:
    # ODE f(t+dt) = f(t) + df/dt * dt
    def euler(self, x_t, v, dt):
        """
        Args:
            x_t: [B, C, H, W] at time t
            v: [B, C, H, W] velocity (predicted by the model)
            dt: float, time step
        """
        x_t = x_t + v * dt
        return x_t
    
    # x_t = (1-t)x_0 + t*x_1 
    def create_flow(self, x_1, t, x_0 = None):
        """
        Args:
            x_1: [B, C, H, W] at time t=1
            t: [B] time step
            x_0: [B, C, H, W] at time t=0 (optional)
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1) # sample x_0 from N(0, I)
        t = t[:, None, None, None] # [B, 1, 1, 1]
        x_t = (1 - t) * x_0 + t * x_1
        return x_t, x_0
    
    # Loss = MSE(x_1 - x_0 - v(t))
    def mse_loss(self, v_predict, x_t, x_0):
        """
        Args:
            v_predict: [B, C, H, W] predicted velocity by the model
            x_t: [B, C, H, W] at time t
            x_0: [B, C, H, W] at time t=0
        """
        v_target = x_t - x_0
        loss = F.mse_loss(v_predict, v_target)
        return loss
