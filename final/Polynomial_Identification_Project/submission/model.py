class PolyNet(nn.Module):    # nn.Module is parent class
    def __init__(self):
        super(PolyNet, self).__init__()  # calls init of parent class
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=3,
                                              kernel_size=3,
                                              stride=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(3))
#         126
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=9,
                                              kernel_size=3,
                                              stride=1),
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),
                                              nn.BatchNorm2d(9))
#      62   
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=9,
                                              out_channels=9,
                                              kernel_size=1,
                                              stride=1),
                                              nn.ReLU(),
                                              nn.MaxPool2d(2),
                                              nn.BatchNorm2d(9))
    #30
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=9,
                                                  out_channels=4,
                                                  kernel_size=(1)),
                                                  nn.ReLU(),
                                                  nn.MaxPool2d(2),
                                                  nn.BatchNorm2d(4))
        
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=4,
                                                  out_channels=10,
                                                  kernel_size=(1)),
                                                  nn.Tanh(),
                                                  nn.BatchNorm2d(10))
    #15
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=10,
                                                  out_channels=3,
                                                  kernel_size=(4)),
                                                  nn.LeakyReLU(),
                                                  nn.MaxPool2d(2),
                                                  nn.BatchNorm2d(3))

#         self.layer3 = nn.Sequential(nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(6*6*3, 4)
        # ----------------------------------------------
        # implementation needed here
        # ----------------------------------------------

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        """
        Feed forward through network
        Args:
            x - input to the network
            
        Returns "out", which is the network's output
        """

        # ----------------------------------------------
        # implementation needed here
        # ----------------------------------------------

        return out