import torch
import torch.nn as nn
from torchvision import models
import pdb


class Encoder(nn.Module):
    def __init__ (self, hidden_size):
        super(Encoder, self).__init__()
        
        # Load pretrained resnet model
        resnet = models.resnet152(pretrained=True)
        
        # Remove the fully connected layers
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Create our replacement layers
        # We reuse the in_feature size of the resnet fc layer for our first replacement layer = 2048 as of creation
        self.linear = nn.Linear(in_features = resnet.fc.in_features, out_features = hidden_size)
        self.bn = nn.BatchNorm1d(num_features = hidden_size, momentum = 0.01)

    def forward (self, images):
        # Get the expected output from the fully connected layers
        # Fn: AvgPool2d(kernel_size=7, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        # Output: torch.Size([batch_size, 2048, 1, 1])
        features = self.resnet(images)

        # Resize the features for our linear function
        features = features.view(features.size(0), -1)
        
        # Fn: Linear(in_features=2048, out_features=embed_size, bias=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.linear(features)
        
        # Fn: BatchNorm1d(embed_size, eps=1e-05, momentum=0.01, affine=True)
        # Output: torch.Size([batch_size, embed_size])
        features = self.bn(features)
        
        return features

class Classifier(nn.Module):
    def __init__ (self, hidden_size, num_classes=39):
        super(Classifier, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward (self, x):

        out = self.model(x)
        return out

class AdversarialHead(nn.Module):
    def __init__ (self, hidden_size):
        super(AdversarialHead, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
        )

    def forward (self, x):

        out = self.model(x)
        out_detached = self.model(x.detach())
        return (out, out_detached)

class BaselineModel(nn.Module):
    def __init__ (self, hidden_size, num_classes=39):
        super(BaselineModel, self).__init__()

        self.encoder = Encoder(hidden_size)
        self.classifier = Classifier(hidden_size, num_classes)

    def forward (self, images):

        h = self.encoder(images)
        y = self.classifier(h)
        return y, None

    def sample (self, images):
        """
        Method to perform classification without computing
        adversarial head output.
          images: tensor of shape (batch_size, num_channels, height, width)
          return: tensor of shape (batch_size, num_classes)
        """
        h = self.encoder(images)
        y = self.classifier(h)
        return y

class OurModel(nn.Module):
    def __init__ (self, hidden_size, num_classes=39):
        super(OurModel, self).__init__()

        self.encoder = Encoder(hidden_size)
        self.classifier = Classifier(hidden_size, num_classes)
        self.adv_head = AdversarialHead(hidden_size)

    # def forward (self, images, images_subset):
    def forward (self, images, protected_class_labels):

        # h_images = self.encoder(images)
        # y = self.classifer(h_images)
        # h_images_subset = self.encoder(images_subset)
        # a = self.adv_head(h_images_subset)
        h = self.encoder(images) # (batch_size, hidden_size)
        y = self.classifier(h) # (batch_size, num_classes)
        protected_class_encoded_images = h[protected_class_labels] 
        a, a_detached = self.adv_head(protected_class_encoded_images)
        return y, (a, a_detached)

    def sample (self, images):
        """
        Method to perform classification without computing
        adversarial head output.
          images: tensor of shape (batch_size, num_channels, height, width)
          return: tensor of shape (batch_size, num_classes)
        """
        h = self.encoder(images)
        y = self.classifier(h)
        return y


