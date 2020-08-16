import torch
import torch.nn as nn


def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, padding=0):
    """1D convolution with filter size 1 and auto-padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=True)


def batch_norm(out_planes):
    return nn.BatchNorm1d(out_planes)


def activation_layer(layer_type="relu"):
    activation_modules = nn.ModuleDict([
        ["relu", nn.ReLU(inplace=True)],
        ["leaky_relu", nn.LeakyReLU(negative_slope=0.05, inplace=True)],
        ["none", nn.Identity()]
    ])
    return activation_modules[layer_type]


class PaddedGlobalPool1d(nn.AdaptiveMaxPool1d):
    """
    Adaptive Max Pooling for zero padded sequences
    """
    def __init__(self, output_size, pool_type="max", *args, **kwargs):
        super().__init__(output_size, *args, **kwargs)
        self.output_size = output_size
        self.pool_type = pool_type

    def forward(self, x_input, sample_lengths):
        x = x_input.clone()
        resample_factor = max(sample_lengths) / x.size(-1)
        for sample_idx in range(x.size(0)):
            temporal_idx = int(sample_lengths[sample_idx] // resample_factor) - 1
            x[sample_idx, :, temporal_idx:] = float("-Inf")

        if self.pool_type == "max":
            x = nn.functional.adaptive_max_pool1d(x, self.output_size)
        elif self.pool_type == "avg":
            x = nn.functional.adaptive_avg_pool1d(x, self.output_size)
        else:
            pass

        x = x.view([x.size(0), x.size(1) * x.size(2)])
        return x


class ReshapeTensor(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class CustomWrapper(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.content = nn.ModuleList([*args])

    def add(self, item):
        self.content.append(item)

    def forward(self, x, sample_lengths):
        for item in self.content:
            x = item(x, sample_lengths)
        return x


class SuperBlock(nn.Module):
    """
    Super class for building general ResNet blocks.
    Original residual block is replaced by pre-activation variant (No layers after addition).
    """
    def __init__(self, in_planes, out_planes, cardinality):
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.cardinality = cardinality
        self.blocks = nn.Identity()
        self.resample = nn.Identity()
        self.squeeze = None

    def forward(self, x, sample_length):
        # hold and/or resample input data
        if self.has_same_shape:
            identity = x
        else:
            identity = self.resample(x)
        # forward pass through residual block
        x = self.blocks(x)
        # squeze and excite
        if self.squeeze is not None:
            x_pooled = self.pool(x, sample_length)
            x *= self.squeeze(x_pooled)
        # addition
        x += identity
        # no activation here due to preactivation within residual block
        return x

    @property
    def has_same_shape(self):
        """Check for equality of input and output plane dimensions"""
        return self.in_planes == self.out_planes and self.in_planes == self.expanded_channels

    @property
    def expanded_channels(self):
        """Compute correct number of output planes (filters in block)"""
        return self.out_planes * self.expansion


class ResidualBlock(SuperBlock):
    expansion = 1
    """Shortcut with input data reshaping."""
    def __init__(self, in_planes, out_planes, cardinality, expansion=1, downsampling=1,
                 filter_type=conv1x3, activation_type="relu", *args, **kwargs):
        super().__init__(in_planes, out_planes, cardinality, *args, **kwargs)

        self.expansion = expansion
        self.downsampling = downsampling
        self.convolution = filter_type
        self.activation = activation_type
        self.nb_pools = 1

        # filters for reshaping identity data
        if not self.has_same_shape:
            self.resample = nn.Sequential(
                nn.BatchNorm1d(self.in_planes),
                activation_layer(self.activation),
                conv1x1(self.in_planes, self.out_planes, stride=self.downsampling),
            )

        # residual block
        self.blocks = nn.Sequential(
            nn.BatchNorm1d(self.in_planes),
            activation_layer(self.activation),
            self.convolution(self.in_planes, self.in_planes, groups=cardinality),

            nn.BatchNorm1d(self.in_planes),
            activation_layer(self.activation),
            self.convolution(self.in_planes, self.out_planes, stride=self.downsampling),
        )


class SqueezedResidualBlock(ResidualBlock):
    def __init__(self, in_planes, out_planes, cardinality, expansion=1, downsampling=1,
                 filter_type=conv1x3, activation_type="relu", *args, **kwargs):
        super().__init__(in_planes, out_planes, cardinality, expansion=expansion, downsampling=downsampling,
                         filter_type=filter_type, activation_type=activation_type, *args, **kwargs)

        self.nb_pools = 1

        # pooling block
        self.pool = PaddedGlobalPool1d(self.nb_pools, "max")

        # squeeze-excitation block
        self.squeeze = nn.Sequential(
            ReshapeTensor([-1, self.out_planes * self.nb_pools]),
            nn.Linear(self.out_planes * self.nb_pools, self.out_planes, bias=True),
            nn.Sigmoid(),
            ReshapeTensor([-1, self.out_planes, 1]),
        )


class BasicLayer(nn.Module):
    """
    A ResNet layer with N stacked blocks.
    """
    def __init__(self, in_planes, out_planes, cardinality=1, block_type=ResidualBlock, depth=1, *args, **kwargs):
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.cardinality = cardinality

        if self.has_same_shape:
            self.downsampling = 1
        else:
            self.downsampling = 2

        self.layer = CustomWrapper()

        # stack blocks
        for _ in range(1, depth):
            self.layer.add(
                block_type(self.in_planes, self.in_planes, self.cardinality,
                           downsampling=1, *args, **kwargs)
            )

        # stack first block with downsampling
        self.layer.add(
            block_type(self.in_planes, self.out_planes, self.cardinality, *args, **kwargs,
                       downsampling=self.downsampling)
        )

    def forward(self, x, sample_length):
        x = self.layer(x, sample_length)
        return x

    @property
    def has_same_shape(self):
        """Check for equality of input and output plane dimensions"""
        return self.in_planes == self.out_planes


class ResidualLayer(BasicLayer):
    """
    A ResNet layer with N stacked blocks and identity shortcut.
    """
    def __init__(self, in_planes, out_planes, cardinality=1, activation_type="relu",
                 block_type=ResidualBlock, depth=1, *args, **kwargs):
        super().__init__(in_planes, out_planes, cardinality=cardinality, block_type=block_type,
                         depth=depth, *args, **kwargs)

        self.activation = activation_type

        # set filters for reshaping identity data
        if not self.has_same_shape:
            self.resample = nn.Sequential(
                nn.BatchNorm1d(self.in_planes),
                activation_layer(self.activation),
                conv1x1(self.in_planes, self.out_planes, stride=self.downsampling),
            )

    def forward(self, x, sample_length):
        # hold and/or resample input data
        if self.has_same_shape:
            identity = x
        else:
            identity = self.resample(x)
        # forward pass through residual layer
        x = self.layer(x, sample_length)
        # addition
        x += identity
        return x


class ResNet(nn.Module):

    def __init__(self, num_classes, in_planes, layer_planes=[64, 128, 256, 512], layer_depths=[2, 2, 2, 2],
                 layer_cardinality=[16, 16, 16, 16], block_type=ResidualBlock, layer_type=ResidualLayer,
                 activation_type="relu", *args, ** kwargs
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.layer_planes = layer_planes
        self.layer_depths = layer_depths
        self.layer_cardinality = layer_cardinality

        # input gate
        self.gate = nn.Sequential(
            conv1x1(in_planes, self.layer_planes[0], stride=2),
            nn.BatchNorm1d(self.layer_planes[0]),
            activation_layer(activation_type),
        )

        # Generate ResNet layers
        layer_params = zip(self.layer_planes[:-1], self.layer_planes[1:], self.layer_depths, self.layer_cardinality)

        self.features = CustomWrapper()
        for in_planes, out_planes, depth, cardinality in layer_params:
            self.features.add(
                layer_type(in_planes*block_type.expansion, out_planes, cardinality=cardinality, depth=depth,
                           activation_type=activation_type, block_type=block_type, *args, **kwargs),
            )

        # unpad layer
        self.global_max_pool = PaddedGlobalPool1d(1, "max")

        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.layer_planes[-1], self.num_classes, bias=True),
            nn.Sigmoid()
        )

        # inititalization
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, sample_lengths):
        x = self.gate(x)
        x = self.features(x, sample_lengths)
        x = self.global_max_pool(x, sample_lengths)
        x = self.classifier(x)
        return x


def main():
    batch_size, in_planes, sample_length = 8, 12, 15000
    dummy = torch.rand((batch_size, in_planes, sample_length))
    lens = [7500, 15000, 12300, 15000, 13000, 9000, 5000, 4300]

    model_params = {
        "block_type": SqueezedResidualBlock,
        "layer_type": ResidualLayer,
        "activation_type": "leaky_relu",
        "layer_planes": [32, 64, 96, 128, 160, 192],
        "layer_depths": [2, 2, 2, 2, 2, 2],
        "layer_cardinality": [8, 16, 16, 32, 32, 32],
    }

    model = ResNet(num_classes=9, in_planes=in_planes, **model_params)

    y = model(dummy, sample_lengths=lens)
    print(y)


if __name__ == "__main__":
    main()