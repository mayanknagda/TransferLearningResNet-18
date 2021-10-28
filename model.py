import torchvision.models as models
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.in_feature_size = self.resnet18.fc.in_features
        self.out_features_size = args.out_features_size
        self.resnet18.fc = nn.Linear(self.in_feature_size, self.out_features_size)
        
        if args.config_type == 'a':
            # freeze all layers excepts first layer
            self.config_type_a()
            
        if args.config_type == 'b':
            # freeze all layers excepts second layer
            self.config_type_b()
        
        if args.config_type == 'c':
            # freeze all layers excepts third layer
            self.config_type_c()
        
        if args.config_type == 'd':
            # freeze all layers excepts four layer
            self.config_type_d()
        
        if args.config_type == 'e':
            # train using all layers
            self.config_type_e()
    
    def config_type_a(self):
        # lets freeze the layers
        # freezing layer 2
        for params in self.resnet18.layer2.parameters():
            params.requires_grad = False
        # freezing layer 3
        for params in self.resnet18.layer3.parameters():
            params.requires_grad = False
        # freezing layer 4
        for params in self.resnet18.layer4.parameters():
            params.requires_grad = False
            
    def config_type_b(self):
        # lets freeze the layers
        # freezing layer 1
        for params in self.resnet18.layer1.parameters():
            params.requires_grad = False
        # freezing layer 3
        for params in self.resnet18.layer3.parameters():
            params.requires_grad = False
        # freezing layer 4
        for params in self.resnet18.layer4.parameters():
            params.requires_grad = False
            
    def config_type_c(self):
        # lets freeze the layers
        # freezing layer 1
        for params in self.resnet18.layer1.parameters():
            params.requires_grad = False
        # freezing layer 2
        for params in self.resnet18.layer2.parameters():
            params.requires_grad = False
        # freezing layer 4
        for params in self.resnet18.layer4.parameters():
            params.requires_grad = False
            
    def config_type_d(self):
        # lets freeze the layers
        # freezing layer 1
        for params in self.resnet18.layer1.parameters():
            params.requires_grad = False
        # freezing layer 2
        for params in self.resnet18.layer2.parameters():
            params.requires_grad = False
        # freezing layer 3
        for params in self.resnet18.layer3.parameters():
            params.requires_grad = False
            
    def config_type_e(self):
        # we have to train using all layers
        pass
        
    def forward(self, x):
        out_resnet_18 = self.resnet18(x)
        return out_resnet_18