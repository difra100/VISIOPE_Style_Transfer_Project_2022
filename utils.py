import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_images(models, test, path_content, path_style, model_name, device, figsize):
  c = 0
  for model in models:
    model.to(torch.device(device))
    f, axarr = plt.subplots(len(path_content)+1,len(path_style)+1,figsize=figsize) 
    title = 'Style transfer results with ' + str(model_name[c]) + ' and a style percentage of the '+ str(test*100)+ '%'
    f.suptitle(title, fontdict = {'fontsize': 20})
    
    model.to(torch.device(device))
    for ax1 in axarr.flatten():
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for k in range(len(path_content)+1):
      for j in range(len(path_style)+1):
        if k == 0 and j==0:
          axarr[k,j].set_axis_off()
          continue 
        if k == 0 and j!=0:
          axarr[k,j].imshow(torchvision.transforms.ToPILImage()(path_style[j-1][0]))
          axarr[k,j].set_axis_off()
          continue
        if k != 0 and j==0:
          axarr[k,j].imshow(torchvision.transforms.ToPILImage()(path_content[k-1][0]))
          axarr[k,j].set_axis_off()
          continue
        else:
          axarr[k,j].imshow(torchvision.transforms.ToPILImage()(model(path_content[k-1][0].to(torch.device(device)).unsqueeze(0),path_style[j-1][0].to(torch.device(device)).unsqueeze(0), test).squeeze(0)))
          axarr[k,j].set_axis_off()
          continue
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0.05, hspace=0.05)
    plt.show()
    c+=1

def plot_different_styles(models, content, style, model_name, device, figsize):
  d = 0
  f, axarr = plt.subplots(1,2,figsize=(9, 9)) 
  for ax1 in axarr.flatten():
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_aspect('equal')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
  axarr[0].imshow(torchvision.transforms.ToPILImage()(content))
  axarr[0].set_axis_off()
  axarr[0].set_title('Content image')
  axarr[1].imshow(torchvision.transforms.ToPILImage()(style))
  axarr[1].set_axis_off()
  axarr[1].set_title('Style image')
  plt.show()
  for model in models:
    model.to(torch.device(device))
    f1, axarr1 = plt.subplots(1,11,figsize=figsize)
    for ax1 in axarr.flatten():
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
    c = 0
    for k in range(0,11):
      result = model(content.to(torch.device(device)).unsqueeze(0),style.to(torch.device(device)).unsqueeze(0), c).squeeze(0)
      axarr1[k].imshow(torchvision.transforms.ToPILImage()(result))
      axarr1[k].set_axis_off()
      title = 'Style P. = ' + str(round(c*100,2)) + '%' + ' ('+ model_name[d]+')'
      axarr1[k].set_title(title)
      c+=0.1
    d+=1

def save_model(checkpoint, path):

  torch.save(checkpoint, path)

def load_model(path,model, device):

  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint['model_state']) 
  
  return checkpoint  

