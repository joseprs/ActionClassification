import torch


class ActionClassifier(torch.nn.Module):
    def __init__(self, input_size=128, window_size_sec=20, frame_rate=8, num_classes=17):  # add type of pool
        # TODO: add if max or avg
        # PARAMETERS
        super(ActionClassifier, self).__init__()
        self.input_size = input_size
        self.window_size_sec = window_size_sec
        self.frame_rate = frame_rate
        self.num_classes = num_classes
        self.frames_per_window = (self.frame_rate * self.window_size_sec) + 1

        # Neural Network
        self.pool_layer = torch.nn.MaxPool1d(self.frames_per_window, stride=1)
        self.fc = torch.nn.Linear(self.input_size, self.num_classes)
        # softmax

    def forward(self, x):
        x = self.pool_layer(x.permute(0, 2, 1)).squeeze(-1)
        x = self.fc(x)
        return x
