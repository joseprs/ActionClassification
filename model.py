import torch


class ActionClassifier(torch.nn.Module):
    def __init__(self, pool="AVG", input_size=128, window_size_sec=20, frame_rate=8, num_classes=17):
        # TODO: add if max or avg or NV pool (where do we initialyze the netvlad pool?)
        # PARAMETERS
        super(ActionClassifier, self).__init__()
        self.input_size = input_size
        self.window_size_sec = window_size_sec
        self.frame_rate = frame_rate
        self.num_classes = num_classes
        self.frames_per_window = (self.frame_rate * self.window_size_sec) + 1
        self.pool = pool

        # Neural Network
        if self.pool == "MAX":
            self.pool_layer = torch.nn.MaxPool1d(self.frames_per_window, stride=1)
        elif self.pool == "AVG":
            self.pool_layer = torch.nn.AvgPool1d(self.frames_per_window, stride=1)

        self.fc = torch.nn.Linear(self.input_size, self.num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool_layer(x.permute(0, 2, 1)).squeeze(-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
