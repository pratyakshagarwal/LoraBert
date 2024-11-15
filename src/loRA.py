import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def lora_parameterization(layer, device, rank=1, lora_alpha=1):
    """
    Create a LoRA (Low-Rank Adaptation) parametrization for a given layer.

    Args:
        layer (nn.Module): The layer to which LoRA parametrization will be applied.
        device (str): The device where the layer will reside ('cpu' or 'cuda').
        rank (int): Rank of the low-rank decomposition.
        lora_alpha (float): Scaling factor for LoRA weights.

    Returns:
        LoRAParametrization: The parametrization module for the given layer.
    """
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )

def apply_lora(model, layers, rank=1, device="cpu"):
    """
    Apply LoRA parametrization to specified layers in a model.

    Args:
        model (nn.Module): The model where LoRA will be applied.
        layers (type): The type of layers to target (e.g., nn.Linear).
        rank (int): Rank of the low-rank decomposition.
        device (str): The device where the layers will reside ('cpu' or 'cuda').
    """
    for name, module in model.named_modules():
        if isinstance(module, layers):
            parametrize.register_parametrization(
                module, "weight", lora_parameterization(module, device, rank=rank)
            )

def enable_disable_lora(model, enabled=True):
    """
    Enable or disable LoRA for all parametrized layers in a model.

    Args:
        model (nn.Module): The model where LoRA will be toggled.
        enabled (bool): Whether to enable or disable LoRA weights.
    """
    for name, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            module.parametrizations["weight"][0].enabled = enabled

def freeze_params(model, verbose=False):
    """
    Freeze all model parameters except for LoRA layers and classifier layers.

    Args:
        model (nn.Module): The model whose parameters will be frozen.
        verbose (bool): If True, prints details of frozen parameters.
    """
    for name, param in model.named_parameters():
        if 'lora' not in name and 'classifier' not in name and 'fc' not in name:
            if verbose:
                print(f'Freezing non-LoRA parameter {name}')
            param.requires_grad = False

def check_frozen_parameters(model):
    """
    Print the count of frozen and trainable parameters in the model.

    Args:
        model (nn.Module): The model to inspect.
    """
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
        else:
            frozen_params.append(param)

    print(f"Frozen Parameters: {sum([p.nelement() for p in frozen_params])}")
    print(f"Trainable Parameters: {sum([p.nelement() for p in trainable_params])}")

class LoRAParametrization(nn.Module):
    """
    A LoRA parametrization module for low-rank adaptation of weights.

    Attributes:
        rank (int): Rank of the low-rank decomposition.
        alpha (float): Scaling factor for LoRA weights.
        lora_A (nn.Parameter): The low-rank matrix A.
        lora_B (nn.Parameter): The low-rank matrix B.
        scale (float): Scaling factor derived from alpha and rank.
        enabled (bool): Whether LoRA is enabled.
    """
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        """
        Initialize the LoRAParametrization module.

        Args:
            features_in (int): Number of input features of the original weight matrix.
            features_out (int): Number of output features of the original weight matrix.
            rank (int): Rank of the low-rank decomposition.
            alpha (float): Scaling factor for LoRA weights.
            device (str): The device where LoRA weights will reside.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # Initialize LoRA parameters A and B
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # Scaling factor for the LoRA weights
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        """
        Forward pass with LoRA adaptation.

        Args:
            original_weights (torch.Tensor): The original weight matrix of the layer.

        Returns:
            torch.Tensor: The adapted weight matrix.
        """
        if self.enabled:
            # Adapted weights: W + (B @ A) * scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

if __name__ == '__main__':
    pass
