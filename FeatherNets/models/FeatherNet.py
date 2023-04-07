
# the code base on https://github.com/tonylins/pytorch-mobilenet-v2
import torch.nn as nn
import math
import torch

from torchvision.models.resnet import BasicBlock
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# reference form : https://github.com/moskomule/senet.pytorch  
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
     
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)


class FeatherNet(nn.Module):
    def __init__(self, n_class=2, input_size=224, se = False, avgdown=False, width_mult=1.):
        super(FeatherNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2], # 56x56
            [6, 48, 6, 2], # 14x14
            [6, 64, 3, 2], # 7x7
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
#         building last several layers        
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                  groups=input_channel, bias=False),
                                     )


        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_DW(x)
        
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def FeatherNetA():
    model = FeatherNet(se = True)
    return model

def FeatherNetB():
    model = FeatherNet(se = True,avgdown=True)
    return model


class FeatherNet_Multi(nn.Module):
    def __init__(self, n_class=2, input_size=224, se = False, avgdown=False, width_mult=1.):
        super(FeatherNet_Multi, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2], # 56x56
            [6, 48, 6, 2], # 14x14
            [6, 64, 3, 2], # 7x7
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        """RGB"""
        self.RGB_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.RGB_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.RGB_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.RGB_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.RGB_features = nn.Sequential(*self.RGB_features)

        """Depth"""
        self.DW_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.DW_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.DW_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.DW_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.DW_features = nn.Sequential(*self.DW_features)

        """IR"""
        self.IR_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.IR_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.IR_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.IR_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.IR_features = nn.Sequential(*self.IR_features)


#         building last several layers        
        # self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
        #                           groups=input_channel, bias=False),
        #                              )
        self.final_DW2 = nn.Sequential(nn.Conv2d(input_channel*3, input_channel, kernel_size=3, stride=2, padding=1,
                                  groups=input_channel, bias=False),
                                     )


        self._initialize_weights()

    def forward(self, rgb, ir, depth):
        out1 = self.RGB_features(rgb)
        out2 = self.IR_features(ir)
        out3 = self.DW_features(depth) # torch.Size([1, 64, 7, 7])

        out = torch.cat((out1,out2,out3), dim=1)
        out = self.final_DW2(out)
        
        out = out.view(out.size(0), -1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)   # vanilla convolution

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            # self.theta * out_diff (central difference term)
            return out_normal - self.theta * out_diff

class FeatherNet_Multi_pfus(nn.Module):
    def __init__(self, n_class=2, input_size=224, se = False, avgdown=False, width_mult=1.):
        super(FeatherNet_Multi_pfus, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2], # 56x56
            [6, 48, 6, 2], # 14x14
            [6, 64, 3, 2], # 7x7
        ]

        basic_conv = nn.Conv2d
        basic_conv=Conv2d_cd
        theta = 0.7

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        """RGB"""
        self.RGB_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.RGB_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.RGB_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.RGB_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.RGB_features = nn.Sequential(*self.RGB_features)

        """Depth"""
        self.DW_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.DW_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.DW_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.DW_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.DW_features = nn.Sequential(*self.DW_features)

        """IR"""
        self.IR_features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.IR_features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample))
                else:
                    self.IR_features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample))
                input_channel = output_channel
            if self.se:
                self.IR_features.append(SELayer(input_channel))
        # make it nn.Sequential
        self.IR_features = nn.Sequential(*self.IR_features)


#         building last several layers        
        # self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
        #                           groups=input_channel, bias=False),
        #                              )

        ## feature-level fusion
        # self.final_DW2 = nn.Sequential(nn.Conv2d(input_channel*3, input_channel, kernel_size=3, stride=2, padding=1,
        #                           groups=input_channel, bias=False),
        #                              )    
        
            
        ## pipe-net fusion
        # self.res_0 = self._make_layer(BasicBlock, input_channel*3, 256, 2, stride=2)
        # self.res_1 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)
        # self.res_2 = self._make_layer(BasicBlock, 512, 1024, 2, stride=2)

        ## cdcn fusion        
        self.lastconv2 = nn.Sequential(
            basic_conv(input_channel*3, 256, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(256),
            nn.ReLU(),    
        )
        
        
        self.lastconv3 = nn.Sequential(
            basic_conv(256, 1024, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )

        

        self._initialize_weights()

    def forward(self, rgb, ir, depth):
        out1 = self.RGB_features(rgb)
        out2 = self.IR_features(ir)
        out3 = self.DW_features(depth) # torch.Size([1, 64, 7, 7])

        ## cdcn fusion
        out = torch.cat((out1,out2,out3), dim=1) # torch.Size([1, 192, 7, 7])
        x = self.lastconv2(out)    
        x = self.lastconv3(x) 
        x = F.adaptive_avg_pool2d(x, output_size=1).view(out.size(0), -1)
        return x

        ## pipe-net fusion
        out = torch.cat((out1,out2,out3), dim=1)         
        test_out = self.res_0(out) #; print('res0',x.size())
        test_out = self.res_1(test_out) #; print('res1',x.size())
        test_out = self.res_2(test_out) #; print('res1',x.size())
        test_out = F.adaptive_avg_pool2d(test_out, output_size=1).view(out.size(0), -1)
        return test_out

        ## feature-level fusion
        out = self.final_DW2(out)        
        out = out.view(out.size(0), -1)
        return out

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def FeatherNetModal3():
    # feature-level fusion
    # model = FeatherNet_Multi(se = True,avgdown=True)

    # feature-level fusion
    model = FeatherNet_Multi_pfus(se = True,avgdown=True)
    return model

