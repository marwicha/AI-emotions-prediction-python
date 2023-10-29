import torch
import torchvision
import torch.nn as nn

class VGG(nn.Module):
    """
    Speech Recognition model with a VGG base
    VGG to latent representation (temporal)
    Aggragation of the temporal representation with mean
    Dense-Dense-SoftMax at the head to classify emotions
    """


    def __init__(self, name='vgg11', bn=True, save_path=None):
        """
        :param name: name of the vgg base ('vgg11', 'vgg13', 'vgg16' or 'vgg19')
        :param bn: wether or not to use Batch Normalization
        :param save_path: save path to recover weights of the model if given
        """

        dict_vgg = {
            ('vgg11', False): torchvision.models.vgg11,
            ('vgg11', True): torchvision.models.vgg11_bn,
            ('vgg13', False): torchvision.models.vgg13,
            ('vgg13', True): torchvision.models.vgg13_bn,
            ('vgg16', False): torchvision.models.vgg16,
            ('vgg16', True): torchvision.models.vgg16_bn,
            ('vgg19', False): torchvision.models.vgg19,
            ('vgg19', True): torchvision.models.vgg19_bn,
        }

        super(VGG, self).__init__()
        # set the model and recover pretrained weights of the VGG
        # self.vgg = dict_vgg[(name, bn)](pretrained=True, progress=True)
        self.vgg = dict_vgg[(name, bn)](pretrained=False, progress=True)

        # last convolution to get a Bx1xLxT
        # where B the batch size, L the latent size and T the temporal size
        self.conv_mel_to_flat = nn.Conv2d(
            512,
            512,
            kernel_size=(2, 1),
            stride=1,
            dilation=1,
            padding=0
        )


        # Head from latent representation to emotion classification
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 5)
        self.activation = nn.Softmax()

        # set the device by default GPU if exists
        # and recover saved model if given
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if save_path is not None:
            self.load_state_dict(torch.load(save_path, map_location=self.device))

    def predict_sample(self, x):
        """
        Make a prediction for one MEL Spectrogram input

        :param x: MEL spectrogram
        :return: prediction as emotion array
        """
        # create channel axis 1 x Mel x Time
        x = torch.from_numpy(x).unsqueeze(0)

        # assure the input is at least Time > 32 to avoid computing issues
        minimum_t = 32
        if x.size()[-1] < minimum_t:
            c, m, t = x.size()
            _x = torch.zeros((c, m, minimum_t))
            _x[..., :t] = x
            x = _x

        # set the input to the model's device then add the 1 bacth axis
        x = x.to(self.device)
        pred, latent = self.forward(x.unsqueeze(0))

        # get the prediciton and latent to the current cpu device to numpy
        pred = pred[0].detach().cpu().numpy()
        latent = latent[0].detach().cpu().numpy()
        return pred, latent


    def forward(self, x):
        """
        Forward pass of the model

        :param x: MEL spectrogram
        :return: prediction as emotion array
        """
        # convert Batch x 1 x Mel x Time to Batch x 3 x Mel x Time
        # because the model was trained on RGB images
        x = torch.cat([x, x, x], dim=1)

        # get the features Batch x 3 x Latent x Time
        x = self.vgg.features(x)
        # convert Batch x 3 x Latent x Time to Batch x (1) x Latent x Time
        x = self.conv_mel_to_flat(x).squeeze(-2)  # (Batch x Latent x Time)
        # aggregate the latent space with temporal mean
        x = torch.mean(x, dim=2)  # (Batch x Latent)

        latent = x
        # make classifcation from latent space
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.activation(self.fc2(x))  # (Batch x N_class)
        return x, latent




# model = VGG('vgg11', bn=True)
# model(torch.zeros((3, 1, 80, 447)))
# total_params, trainable_params = summary(model, (1, 80, 447), device='cpu', batch_size=1)
