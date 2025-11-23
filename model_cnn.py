import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, genes, input_channels=3, num_classes=10, input_size=32):
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channels
        current_size = input_size

        for i, conv_config in enumerate(genes['conv_configs']):
            filters = conv_config['filters']
            kernel_size = conv_config['kernel_size']
            padding = kernel_size // 2
            
            layers.append(nn.Conv2d(in_channels, filters, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(filters))

            if genes['activation'] == 'relu':
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.LeakyReLU(0.1, inplace=True))

            if (i + 1) % 2 == 0 or i == len(genes['conv_configs']) - 1:
                if genes['pool_type'] == 'max':
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.append(nn.AvgPool2d(2, 2))
                current_size = current_size // 2
            
            in_channels = filters
        
        self.features = nn.Sequential(*layers)
        self.flat_size = in_channels * current_size * current_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, genes['fc_units']),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(genes['fc_units'], num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x