import torch
from netvlad import NetVLAD, NetRVLAD


class ActionClassifier(torch.nn.Module):
    def __init__(self, pool="MAX", input_size=128, window_size_sec=20, frame_rate=8, num_classes=17):
        # PARAMETERS
        super(ActionClassifier, self).__init__()
        self.input_size = input_size
        self.window_size_sec = window_size_sec
        self.frame_rate = frame_rate
        self.num_classes = num_classes
        self.frames_per_window = (self.frame_rate * self.window_size_sec) + 1
        self.pool = pool
        self.vlad_k = 64

        # Neural Network
        if self.pool == "MAX":
            self.pool_layer = torch.nn.MaxPool1d(self.frames_per_window, stride=1)
            self.fc = torch.nn.Linear(self.input_size, self.num_classes)
        elif self.pool == "AVG":
            self.pool_layer = torch.nn.AvgPool1d(self.frames_per_window, stride=1)
            self.fc = torch.nn.Linear(self.input_size, self.num_classes)
        elif self.pool.lower() == 'netvlad':
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k, feature_size=self.input_size, add_batch_norm=True)
            self.fc = torch.nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)
        elif self.pool.lower() == 'netrvlad':
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k, feature_size=self.input_size, add_batch_norm=True)
            self.fc = torch.nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        if self.pool == 'MAX' or self.pool == 'AVG':
            x = self.pool_layer(x.permute(0, 2, 1)).squeeze(-1)
        elif self.pool.lower() == "netvlad" or self.pool.lower() == "netrvlad":
            x = self.pool_layer(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
