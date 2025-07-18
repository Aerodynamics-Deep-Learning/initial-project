"""

Here, we define the Fourier Layers that are used in the Fourier Neural Operator (FNO) architecture.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# These are directly taken from the paper since it is straightfoward. Perform matmul.
@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
    res = torch.einsum("bim,iom->bom", a, b)
    return res

@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  torch.einsum("bixy,ioxy->boxy", a, b)
    return res


@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


#region Spectral Convolution

class SpectralConvolution1D(nn.Module):
    
    """
    Performs the 1D spectral (Fourier) convolution operation.

    Parameters:
        in_channels is how many inputs we have to our channel,
        out_channels is how many outputs we want,
        modes is the number of Fourier modes to keep in the layer.

    For more detail on how it works, and some charts, refer to the original FNO paper.
    """
    
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConvolution1D, self).__init__()

        self.in_channels = in_channels 
        self.out_channels = out_channels
        # The scaling factor for the randomized weights
        self.scale = 1 / (in_channels * out_channels) 

        # How many modes of the FT we want to keep
        self.modes1 = modes

        # Initialize the weights for in-frequency-space linear transformation, the weights are complex numbers, so we use torch.cfloat
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):

        batchsize = x.shape[0]
        # The first dimension is the batch size, the second is the number of channels, and the third is the signal itself on that channel.

        # Compute the Fourier transform of the input using FFT
        x_ft = torch.fft.rfftn(x, dim=[2]) # Apply fft along the last dimension, the signal dimension. 

        # Define a zeros tensor to hold the outputs.
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:,:,:self.modes1] = compl_mul1d(x_ft[:,:,:self.modes1], self.weights1)

        # Compute the inverse FT using irfftn, "return to physical space"
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])

        return x
    

#endregion


#region Fourier Layer

class FourierLayer1DLocal(nn.Module):

    """
    The complete Fourier Layer, as is on the original FNO paper. Take the input, apply spectral convolution, apply local linear transformation, add these two and that becomes the output.

    Parameters:
        in_channels: How many input channels we have
        out_channels:   

    """

    def __init__(self, in_channels, out_channels, modes, kernel):
        super(FourierLayer1DLocal, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.kernel = kernel

        self.SpectralConv = SpectralConvolution1D(in_channels, out_channels, modes)

        if kernel == 1:
            self.linear = nn.Conv1d(in_channels, out_channels, kernel)
        else: 
            self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel//2)
        
    def forward(self, x):
        """
        input x: (batchsize, channel_width, x_grid)
        """

        x1 = self.SpectralConv(x)
        x2 = self.linear(x)

        return x1 + x2

#endregion