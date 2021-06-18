import gym
import torch as th
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn.modules import linear
from torch.nn.modules.pooling import MaxPool2d
class CustomCNNSimple(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int=256):
        super(CustomCNNSimple, self).__init__(observation_space, features_dim=features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),



        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
        
class ResNetNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int):
        super(ResNetNetwork, self).__init__(observation_space, features_dim=features_dim)
        n_input_channels = observation_space.shape[0]
        hidden_sizes = [18, features_dim, features_dim]
        pretrained_CNN = 'resnet'+str(hidden_sizes[0])
        print('================',pretrained_CNN)
        self.resnet = th.hub.load('pytorch/vision:v0.9.0', pretrained_CNN, pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_sizes[1])
        
        self.net = nn.Sequential(
            self.resnet
        )
        print(self.net)
        

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.net(obs)
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int=64):
        super().__init__(observation_space, features_dim=features_dim)
        n_input_channels = observation_space.shape[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernal_size =5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(4*4*128, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        out  =self.conv1(observations)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(observations), self.value_net(observations)


class CustomNetworkWithResNet(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_sizes = [18, 64, 64],
        last_layer_dim_pi: int =64,
        last_layer_dim_vf: int =64,
        ):
        super(CustomNetworkWithResNet, self).__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        pretrained_CNN = 'resnet'+str(hidden_sizes[0])
        self.resnet = th.hub.load('pytorch/vision:v0.9.0', pretrained_CNN, pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_sizes[1])
        print(self.resnet)
        self.policy_net = nn.Sequential(
            self.resnet
        )

        self.value_net = nn.Sequential(
            self.resnet

        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(obs), self.value_net(obs)

        


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetworkWithResNet(self.features_dim)