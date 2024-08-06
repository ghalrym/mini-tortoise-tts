import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, functional
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def get_padding(k, d):
    return int((k * d - d) / 2)


class ResBlock1(torch.nn.Module):
    """Residual Block Type 1. It has 3 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1_1 -> conv1_2 -> conv1_3 -> z -> lrelu -> conv2_1 -> conv2_2 -> conv2_3 -> o -> + -> o
        |--------------------------------------------------------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
                weight_norm(
                    Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                ),
            ]
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            Tensor: output tensor.
        Shapes:
            x: [B, C, T]
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = functional.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = functional.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """Residual Block Type 2. It has 1 convolutional layers in each convolutional block.

    Network::

        x -> lrelu -> conv1-> -> z -> lrelu -> conv2-> o -> + -> o
        |---------------------------------------------------|


    Args:
        channels (int): number of hidden channels for the convolutional layers.
        kernel_size (int): size of the convolution filter in each layer.
        dilations (list): list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = functional.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class HifiganGenerator(torch.nn.Module):
    __slots__ = (
        "inference_padding",
        "num_kernels",
        "num_upsamples",
        "conv_pre",
        "ups",
        "resblocks",
        "conv_post",
        "cond_layer",
        "device",
        "cond_layer",
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resblock_type: str,
        resblock_dilation_sizes: list[list[int]],
        resblock_kernel_sizes: list[int],
        up_sample_kernel_sizes: list[int],
        up_sample_initial_channel: int,
        up_sample_factors: list[int],
        inference_padding: int = 5,
        cond_channels: int = 0,
        conv_pre_weight_norm: bool = True,
        conv_post_weight_norm: bool = True,
        conv_post_bias: bool = True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(up_sample_factors)

        # initial upsampling layers
        self.conv_pre = weight_norm(Conv1d(in_channels, up_sample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(up_sample_factors, up_sample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        up_sample_initial_channel // (2**i),
                        up_sample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = up_sample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias))
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, up_sample_initial_channel, 1)

        if not conv_pre_weight_norm:
            remove_weight_norm(self.conv_pre)

        if not conv_post_weight_norm:
            remove_weight_norm(self.conv_post)

        self.device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

    def forward(self, x, g=None):
        """
        Args:
            x (Tensor): feature input tensor.
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            o = functional.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
        o = functional.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o

    @torch.no_grad()
    def inference(self, c, g=None):
        """
        Args:
            x (Tensor): conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        # c = c.to(self.conv_pre.weight.device)
        # c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        up_1 = torch.nn.functional.interpolate(
                c.transpose(1,2),
                scale_factor=[1024 / 256],
                mode="linear",
            )
        up_2 = torch.nn.functional.interpolate(
            up_1,
            scale_factor=[24000 / 22050],
            mode="linear",
        )
        g = g.unsqueeze(0)
        return self.forward(up_2.to(self.device), g.transpose(1,2))

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
