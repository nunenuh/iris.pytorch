import torch.nn as nn

class IrisNetwork(nn.Module):
    def __init__(self, in_num, hid_num, out_num):
        super(IrisNetwork, self).__init__()
        self.linear1 = nn.Linear(in_num, hid_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hid_num, hid_num)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(hid_num, out_num)

    def forward(self, input):
        x = self.linear1(input)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
