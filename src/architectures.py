import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Encoder:

  def __init__(self,device):
    self.device = device
    self.vgg19_norm = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True), # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True), # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1), # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True), # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True)
            ).to(device)
          
    self.vgg19_norm.load_state_dict(torch.load("./models/vgg_normalised.pth"))

    

  def forward(self, input, lista = False):
    out = input
    layer_list = [3, 10, 17, 30] # These are the positions of the interest layer (where we want to compute the loss).
    tensor_list = []
    for n_layer,layer in enumerate(self.vgg19_norm):
      out = layer(out)
      if n_layer in layer_list:
        tensor_list.append(out)
        if n_layer == layer_list[-1]:
          break  
    if lista:
      return tensor_list

    return out 

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.padding = nn.ReflectionPad2d(padding=1) # Using reflection padding as described in vgg19
    self.US = nn.Upsample(scale_factor=2, mode="nearest")

    self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

    self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

    self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

    self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


  def forward(self, x):
    out = self.US(F.relu(self.conv4_1(self.padding(x))))

    out = F.relu(self.conv3_1(self.padding(out)))
    out = F.relu(self.conv3_2(self.padding(out)))
    out = F.relu(self.conv3_3(self.padding(out)))
    out = self.US(F.relu(self.conv3_4(self.padding(out))))

    out = F.relu(self.conv2_1(self.padding(out)))
    out = self.US(F.relu(self.conv2_2(self.padding(out))))
    out = F.relu(self.conv1_1(self.padding(out)))
    out = self.conv1_2(self.padding(out))
    return out


class DecodedRes(nn.Module):
  def __init__(self, residuals):
    super(DecodedRes,self).__init__()
    self.pad = nn.ReflectionPad2d(padding=1)
    self.US = nn.Upsample(scale_factor = 2, mode = 'nearest') 
    self.r = residuals # This is a boolean that inactivate the residual block if set to false.
    self.conv5_1 = nn.Conv2d(512,512,kernel_size = (3,3), padding = 0, bias = True)
    self.conv5_2 = nn.Conv2d(512,512,kernel_size = (3,3), padding = 0, bias = True)
    
    if self.r: # These are the residual blocks where should be the convolution.
      self.conv4_1_2 = nn.Conv2d(512,256,kernel_size = (3,3), padding = 0, bias = True)
      self.conv3_0_1 = nn.Conv2d(256,128,kernel_size = (3,3), padding = 0, bias = True)
      self.conv3_1_2 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)
      self.conv2_0_1 = nn.Conv2d(128,64,kernel_size = (3,3), padding = 0, bias = True)


    self.conv4_1 = nn.Conv2d(512,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_2 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv4_3 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_4 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv4_5 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_6 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv4_7 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_8 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv4_9 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_10 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv4_11 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    self.conv4_12 = nn.Conv2d(256,256,kernel_size = (3,3), padding = 0, bias = True)
    
    
    self.conv3_1 = nn.Conv2d(256,128,kernel_size = (3,3), padding = 0, bias = True)
    self.conv3_2 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv3_3 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)
    self.conv3_4 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)
    
    
    self.conv3_5 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)
    self.conv3_6 = nn.Conv2d(128,128,kernel_size = (3,3), padding = 0, bias = True)

    self.conv2_1 = nn.Conv2d(128,64,kernel_size = (3,3), padding = 0, bias = True)
    self.conv2_2 = nn.Conv2d(64,64,kernel_size = (3,3), padding = 0, bias = True)
    
    self.conv2_3 = nn.Conv2d(64,64,kernel_size = (3,3), padding = 0, bias = True)
    self.conv2_4 = nn.Conv2d(64,64,kernel_size = (3,3), padding = 0, bias = True)

    self.conv1 = nn.Conv2d(64,3, kernel_size = (3,3),padding = 0, bias = True)

  def forward(self, input):
    
    out = F.relu(self.pad(self.conv5_1(input)))
    out = self.US(F.relu(self.pad(self.conv5_2(out))))
    
    
    out1 = F.relu(self.pad(self.conv4_1(out)))
    if self.r:
      res1 = out
      res1 = F.relu(self.pad(self.conv4_1_2(res1)))
      out = F.relu(self.pad(self.conv4_2(out1))) + res1
    else:
      out = F.relu(self.pad(self.conv4_2(out1)))
    
    out1 = F.relu(self.pad(self.conv4_3(out)))
    if self.r:
      res2 = out
      out = F.relu(self.pad(self.conv4_4(out1))) + res2
    else:
      out = F.relu(self.pad(self.conv4_4(out1)))
    
    out1 = F.relu(self.pad(self.conv4_5(out)))
    
    if self.r:
      res3 = out
      out = F.relu(self.pad(self.conv4_6(out1))) + res3
    else:
      out = F.relu(self.pad(self.conv4_6(out1)))
    
    out1 = F.relu(self.pad(self.conv4_7(out)))
    if self.r:
      res4 = out
      out = F.relu(self.pad(self.conv4_8(out1))) + res4
    else:
      out = F.relu(self.pad(self.conv4_8(out1)))
    
    out1 = F.relu(self.pad(self.conv4_9(out)))
    
    if self.r:
      res5 = out
      out = F.relu(self.pad(self.conv4_10(out1))) + res5
    else:
      out = F.relu(self.pad(self.conv4_10(out1)))


    out1 = F.relu(self.pad(self.conv4_11(out)))
    out1 = F.relu(self.pad(self.conv4_12(out1)))

    if self.r:
      res6 = out 
      res6 = F.relu(self.pad(self.conv3_0_1(res6)))
      out = self.US(F.relu(self.pad(self.conv3_1(out1))) + res6)
    else:
      out = self.US(F.relu(self.pad(self.conv3_1(out1))))

    if self.r:
      res7 = out
      out = F.relu(self.pad(self.conv3_2(out))) + res7
    else:
      out = F.relu(self.pad(self.conv3_2(out)))
    
    out1 = F.relu(self.pad(self.conv3_3(out)))
    
    if self.r:
      res8 = out
      out = F.relu(self.pad(self.conv3_4(out1))) + res8
    else:
      out = F.relu(self.pad(self.conv3_4(out1)))

    out1 = F.relu(self.pad(self.conv3_5(out)))
    
    if self.r:
      res9 = out
      out = self.US(F.relu(self.pad(self.conv3_6(out1))) + res9)
    else:
      out = self.US(F.relu(self.pad(self.conv3_6(out1))))

    out1 = F.relu(self.pad(self.conv2_1(out)))

    if self.r:
      res10 = out
      res10 = F.relu(self.pad(self.conv2_0_1(res10)))
      out = F.relu(self.pad(self.conv2_2(out1))) +res10
    else:
      out = F.relu(self.pad(self.conv2_2(out1)))

    out1 = F.relu(self.pad(self.conv2_3(out)))
    
    if self.r:  
      res11 = out
      out = F.relu(self.pad(self.conv2_4(out1))) +res11
    else:
      out = F.relu(self.pad(self.conv2_4(out1)))

    out = self.pad(self.conv1(out))
    
    

    return out

