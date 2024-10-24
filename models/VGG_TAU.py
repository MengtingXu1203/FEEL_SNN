from models.layers import *
import math

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ],
    'vggdvs': [
        [64, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A'],
        []
    ],
    'vgggesture': [
        [16, 32, 'A'],
        [32, 32, 'A'],
        [],
        [],
        []
    ]
}

class VGG_TAU(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, tau=1., init_c=3, args=None):
        super(VGG_TAU, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.tau = tau
        self.init_channels = init_c
        self.args = args

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16
            #self.W = 49 ##only for dct
        else:
            self.W = 1
        

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [64,32,32]),
            nn.AvgPool2d(2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [128,16,16])
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [256,16,16]),
            nn.AvgPool2d(2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [512,8,8])
            )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [512,8,8])
            )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [512,8,8]),
            nn.AvgPool2d(2)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [512,4,4])
            )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            LIFSpikeTau(T=self.T, tau=self.tau, shape = [512,4,4])
            )
        self.layer9 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*self.W, 4096),
            LIFSpikeTau(self.T, tau=self.tau, shape = [4096]),
            nn.Linear(4096, 4096),
            LIFSpikeTau(self.T, tau=self.tau, shape = [4096]), 
            nn.Linear(4096, num_class)
            )
        
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)
        self.hooked = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def hook(self, model, gradin, gradout):
        self.hooked = gradin[0]

    def act_hook(self, model, input, output):
        x = input[0]
        if self.T > 0:
            x = self.expand(x)
            x = x.mean(0)
        self.hooked = x
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input, freq_filter=None):
        
        if self.T > 0:
            if self.args.encode == 'ft':
                input = ft(input, freq_filter)

            else:
                input = add_dimention(input, self.T)
            input = self.merge(input)
        out = self.layer1(input)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        if self.T > 0:
            out = self.expand(out)
            out = out.permute(1,0,2) 

        return out

