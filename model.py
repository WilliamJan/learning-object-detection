import torch
import torch.nn as nn

architecture_config = [
    #(kernel_size, num_filter, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        return self.fcs(torch.flatten(self.darknet(x), start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for i in architecture:
            if type(i) == tuple:
                layers += [
                    CNNBlock(in_channels, i[1], kernel_size=i[0], stride=i[2], padding=i[3])
                ]
                in_channels = i[1]
            elif type(i) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(i) == list:
                for _ in range(i[-1]):
                    for j in i[:-1]:
                        layers += [
                            CNNBlock(in_channels, j[1], kernel_size=j[0], stride=j[2], padding=j[3])
                        ]
                        in_channels = j[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 496), #4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5))

        )

model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
x = torch.randn((2, 3, 448, 448))
print(model(x).shape)