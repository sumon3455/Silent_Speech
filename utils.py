# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import pandas as pd
import os
from skimage import io
import cv2
# Image manipulations
from PIL import Image
# Timing utility
from timeit import default_timer as timer 
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import nibabel as nib


class MyData(Dataset):
    
    def __init__(self, root_dir,categories,img_names,target,my_transforms, return_path,ONN,mean,std):
        self.root_dir = root_dir
        self.categories = categories
        self.img_names = img_names
        self.target = target
        self.my_transforms = my_transforms
        self.return_path = return_path
        self.ONN = ONN
        self.mean  = mean 
        self.std  = std 
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
      
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        # read image
        y = self.target[index].squeeze()
        label = y.item()
        x = nib.load(os.path.join(self.root_dir, self.categories[label]+'/'+self.img_names[index]))
        x = x.get_fdata()
        x = torch.from_numpy(x)
        x = x.permute(0,1,3,2)
        x = x.permute(0,2,1,3)
        x = x/255.0 
        x = x.type(torch.FloatTensor)
        # apply transformation
        # x = self.my_transforms(x) 
        if self.ONN:
            # x = x-self.mean / self.std 
            x = 2.0*x -1  
        if self.return_path: 
            return x, y,  self.categories[label]+'/'+self.img_names[index] 
        else:
            return x, y

class MyVideosData(Dataset):
    
    def __init__(self, root_dir,categories,img_names,target,my_transforms, return_path,ONN,mean,std,width,height):
        self.root_dir = root_dir
        self.categories = categories
        self.img_names = img_names
        self.target = target
        self.my_transforms = my_transforms
        self.return_path = return_path
        self.ONN = ONN
        self.mean  = mean 
        self.std  = std 
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std) 
        self.height = height
        self.width = width
      
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        # read image
        y = self.target[index].squeeze()
        label = y.item()
        filename = os.path.join(self.root_dir, self.categories[label]+'/'+self.img_names[index])
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        for fn in range(v_len):
            success, frame = v_cap.read()
            if success is False:
                continue
            # resize frame
            frame = cv2.resize(frame, (self.width,self.height), interpolation = cv2.INTER_CUBIC) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            if i==0:
                x = torch.from_numpy(frame).unsqueeze(0)
                i = i+1
            else:
                x = torch.cat((x, torch.from_numpy(frame).unsqueeze(0)))
        x = x.permute(0,1,3,2)
        x = x.permute(0,2,1,3)
        x = x/255.0 
        x = x.type(torch.FloatTensor)
        # apply transformation
        # x = self.my_transforms(x) 
        if self.ONN:
            # x = x-self.mean / self.std 
            x = 2.0*x -1  
        if self.return_path: 
            return x, y,  self.categories[label]+'/'+self.img_names[index] 
        else:
            return x, y
 
def Createlabels(datadir):
    categories = []
    n_Class = []
    img_names = []
    labels = []
    i = 0
    class_to_idx = {}
    idx_to_class = {}
    for d in os.listdir(datadir): 
        class_to_idx[d] = i
        idx_to_class[i] = d  
        categories.append(d)
        temp = os.listdir(datadir + d)
        img_names.extend(temp)
        n_temp = len(temp)
        if i==0:
            labels = np.zeros((n_temp,1)) 
        else:
            labels = np.concatenate( (labels, i*np.ones((n_temp,1))) )
        i = i+1
        n_Class.append(n_temp)

    return categories, n_Class, img_names, labels,  class_to_idx, idx_to_class


def Retrievelabel(datadir,categories):
    n_Class = []
    img_names = [] 
    labels = [] 
    for i,d in enumerate(categories): 
        temp = os.listdir(datadir + d)
        img_names.extend(temp)
        n_temp = len(temp)
        if i==0:
            labels = np.zeros((n_temp,1)) 
        else:
            labels = np.concatenate( (labels, i*np.ones((n_temp,1))) )
        n_Class.append(n_temp)

    return n_Class, img_names, labels
    

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data 
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def train(model_to_load,
          model,
          stop_criteria,
          criterion,
          optimizer,
          scheduler,
          train_loader,
          valid_loader,
          test_loader,
          save_file_name,
          train_on_gpu,
          history=[],
          max_epochs_stop=5,
          n_epochs=30,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    valid_best_acc = 0 
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    # Main loop
    for epoch in range(n_epochs):
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        train_acc = 0
        valid_acc = 0
        test_acc = 0
        # Set to training
        model.train()
        start = timer() 
        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)   
            # Compute the Loss 
            if model_to_load=='inception_v3': 
                loss1 = criterion(output[0], target)
                loss2 = criterion(output[1], target)
                loss = loss1 + 0.4*loss2
                output = output[0] 
            else:
                loss = criterion(output, target) 
            # loss = criterion(output, to_one_hot(target).to('cuda'))  # use it with mse loss 
            # backpropagation of gradients  
            loss.backward() 
            # Update the parameters
            optimizer.step() 
            # # OneCycle schedulaer step
            # scheduler.step()
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            # Track training progress
            # print( 
            #     f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #     end='\r')
            # release memeory (delete variables)
            del output, data, target 
            del loss, accuracy, pred, correct_tensor 
        # After training loops ends, start validation
        else:
            model.epochs += 1
            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()
                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                        # data, target = data.cuda(), target.cuda()
                    # Forward pass
                    output = model(data)
                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)
                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)
                # test loop
                for data, target, _ in test_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                        # data, target = data.cuda(), target.cuda()
                    # Forward pass
                    output = model(data)
                    # test loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    test_loss += loss.item() * data.size(0)
                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    test_acc += accuracy.item() * data.size(0)
                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                test_loss = test_loss / len(test_loader.dataset)
                scheduler.step(valid_loss) 
                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)
                test_acc = test_acc / len(test_loader.dataset)
                #
                history.append([train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc]) 
                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print( 
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \t Test Loss: {test_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}% \t Validation Accuracy: {100 * valid_acc:.2f}% \t Test Accuracy: {100 * test_acc:.2f}%'
                    )
                # release memeory (delete variables) 
                del output, data, target
                del loss, accuracy, pred, correct_tensor 
                
                if stop_criteria == 'loss': 
                    # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                    # if 1:
                        # Save model 
                        torch.save(model.state_dict(), save_file_name)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch
                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                            )

                            # Load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'val_loss', 'test_loss',
                                    'train_acc','val_acc', 'test_acc'
                                ])
                            return model, history
                elif stop_criteria == 'accuracy': 
                    if valid_acc > valid_best_acc:
                    # if 1:
                        # Save model
                        torch.save(model.state_dict(), save_file_name)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch
                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                            )

                            # Load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[ 
                                    'train_loss', 'val_loss', 'test_loss',
                                    'train_acc','val_acc', 'test_acc'
                                ])
                            return model, history
                
    # Load the best state dict
    model.load_state_dict(torch.load(save_file_name)) 
    # Attach the optimizer 
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'test_loss', 'train_acc', 'val_acc', 'test_acc'])
    return model, history








