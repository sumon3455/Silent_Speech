# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import os
from os import path
from importlib import import_module
import shutil
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# customized functions 
from utils import *
from models import *
# reproducability and deterministic algorithims
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# torch.manual_seed(0)

# Parse command line arguments
fname = "config.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config


class EncoderModel(nn.Module):
    def __init__(self, old_model):
        super(EncoderModel, self).__init__()
        self.old_model = old_model 
        self.new_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), 
            nn.Flatten(), 
            nn.Linear(512, class_num), # ResNet18
            # nn.Linear(2048, class_num), # ResNet50
            nn.LogSoftmax(dim=1)) 
        self.old_model = self.old_model.to('cuda') 
        self.new_model = self.new_model.to('cuda')

    def forward(self, x):
        x = self.old_model(x)  
        x = self.new_model(x[5]) # ResNet18
        return x


if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    input_type  = config['input_type']                  # 'NIFTI' or 'video'
    ImageNet = config['ImageNet']                       # set to 'True' to use ImageNet weights or set to 'False' to train from scratch
    q_order = config['q_order']                         # qth order Maclaurin approximation, common values: {1,3,5,7,9}. q=1 is equivalent to conventional CNN
    ONN = config['ONN']                                    # set to 'True' if you are using ONN
    input_ch = config['input_ch']                       # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    frames = config['frames'] = 1                       # 
    freeze_CNN = config['freeze_CNN']                   # set to true to freeze CNN layers
    LSTM_hidden_size = config['LSTM_hidden_size']       # 
    LSTM_num_layers = config['LSTM_num_layers']         # 
    LSTM_drop_rate = config['LSTM_drop_rate']         # 
    bidirectional = config['bidirectional']
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    input_mean = config['input_mean']                   # Dataset mean
    input_std = config['input_std']                     # Dataset std
    optim_fc = config['optim_fc']                       # 'Adam' or 'SGD'
    lr =  config['lr']                                  # learning rate
    stop_criteria = config['stop_criteria']             # Stopping criteria: 'loss' or 'Accuracy'
    n_epochs= config['n_epochs']                        # number of training epochs
    epochs_patience= config['epochs_patience']          # if val loss did not decrease for a number of epochs then decrease learning rate by a factor of lr_factor
    lr_factor= config['lr_factor']  
    max_epochs_stop = config['max_epochs_stop']         # maximum number of epochs with no improvement in validation loss for early stopping
    num_folds = config['num_folds']                     # number of cross validation folds
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_model = config['load_model']                   # specify path of pretrained model wieghts or set to False to train from scratch                        
    model_name= config['model_name']                    # choose a unique name for result folder            
    model_to_load = config['model_to_load']             # chosse one of the models specified in config file
    fold_to_run = config['fold_to_run'] 
    encoder = config['encoder']                         # set to 'True' if you retrain Seg. model encoder as a classifer
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path 
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    ################## 
    traindir = parentdir + 'Data/Train/'
    testdir =  parentdir + 'Data/Test/'
    valdir =  parentdir + 'Data/Val/'
    # Create  Directory 
    if path.exists(Results_path): 
        pass
    else:
        os.mkdir(Results_path)
    # Create  Directory
    if path.exists(save_path):
        pass
    else:
        os.mkdir(save_path) 
    shutil.copy('/content/config.py', save_path+'/'+'config.py')

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False 


    # loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds+1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1]+1

    for fold_idx in range(loop_start, loop_end):
        print('#############################################################')
        if fold_idx==loop_start:
            print('training using '+model_to_load+' network')
        print(f'started fold {fold_idx}')
        save_file_name = save_path + '/' + model_name + f'_fold_{fold_idx}.pt'
        checkpoint_name = save_path + f'/checkpoint_fold_{fold_idx}.pt'
        # checkpoint_name = save_path + f'/checkpoint.pt'
        traindir_fold = traindir + f'fold_{fold_idx}/'
        testdir_fold = testdir + f'fold_{fold_idx}/' 
        valdir_fold = valdir + f'fold_{fold_idx}/' 


        # Create train labels
        categories, n_Class_train, img_names_train, labels_train, class_to_idx, idx_to_class = Createlabels(traindir_fold)   
        labels_train = torch.from_numpy(labels_train).to(torch.int64) 
        class_num = len(categories)
        # Create val labels
        _, n_Class_val, img_names_val, labels_val, _, _ = Createlabels(valdir_fold)   
        labels_val = torch.from_numpy(labels_val).to(torch.int64) 
        # Create test labels
        _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)   
        labels_test = torch.from_numpy(labels_test).to(torch.int64) 

 
        # Image Dataloader
        # Image Transformation 
        if ONN:
            my_transforms = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                ])   
        else:
            my_transforms = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize((Resize_h,Resize_w),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=input_mean, std=input_std)
                ])   

        # train dataloader 
        if input_type=='NIFTI': 
            train_ds = MyData(root_dir=traindir_fold,categories=categories,img_names=img_names_train,target=labels_train,my_transforms=my_transforms,return_path=False,ONN=ONN,mean=input_mean,std=input_std)
        elif input_type=='video':
            train_ds = MyVideosData(root_dir=traindir_fold,categories=categories,img_names=img_names_train,target=labels_train,my_transforms=my_transforms,return_path=False,ONN=ONN,mean=input_mean,std=input_std,width=Resize_w,height=Resize_h)   
        if (len(train_ds)/batch_size)==0:
            train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=1) 
        else:
            train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=1,drop_last=True)  
        # validation dataloader 
        if input_type=='NIFTI': 
            val_ds = MyData(root_dir=valdir_fold,categories=categories,img_names=img_names_val,target=labels_val,my_transforms=my_transforms,return_path=False,ONN=ONN,mean=input_mean,std=input_std)
        elif input_type=='video':
            val_ds = MyVideosData(root_dir=valdir_fold,categories=categories,img_names=img_names_val,target=labels_val,my_transforms=my_transforms,return_path=False,ONN=ONN,mean=input_mean,std=input_std,width=Resize_w,height=Resize_h) 
        val_dl = DataLoader(val_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
        # test dataloader
        if input_type=='NIFTI': 
            test_ds = MyData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std)
        elif input_type=='video':
            test_ds = MyVideosData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std,width=Resize_w,height=Resize_h) 
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)


        # release memeory (delete variables)
        del  n_Class_train, img_names_train, labels_train
        del  n_Class_val, img_names_val, labels_val 

        # load model
        if load_model: 
            checkpoint = torch.load(load_model)
            model = checkpoint['model']  
            del checkpoint 
            # fine tunning encoder of segmentation models
            if encoder:
                # model = model.encoder.features 
                model = EncoderModel(model.encoder)  # ResNet
                model = model.to('cuda') 
        else: 
            baseModel, num_features  = get_pretrained_model(parentdir, model_to_load, freeze_CNN, ImageNet,input_ch,class_num,q_order) 
            model = CNN_LSTM(baseModel, num_features, LSTM_hidden_size, LSTM_num_layers, LSTM_drop_rate,bidirectional, class_num,q_order)

        # Move to gpu and parallelize
        if train_on_gpu:
            model = model.to('cuda')
        # if multi_gpu:
        #     model = nn.DataParallel(model)

        
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')
    
        # choose model loss function and optimizer 
        criterion = nn.NLLLoss() 
        if optim_fc == 'Adam':  
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 
        elif optim_fc == 'SGD': 
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False)   

        ## OneCycle schedulaer
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dl), epochs=n_epochs)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=epochs_patience, verbose=True, 
            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08) 
        # scheduler =[] 

        # Training
        model, history = train(
            model_to_load,
            model, 
            stop_criteria,
            criterion,
            optimizer,
            scheduler, 
            train_dl,
            val_dl,
            test_dl,
            checkpoint_name,
            train_on_gpu,
            history=[],
            max_epochs_stop=max_epochs_stop,
            n_epochs=n_epochs,
            print_every=1)

        # # Saving TrainModel
        TrainChPoint = {} 
        TrainChPoint['model']=model                              
        TrainChPoint['history']=history
        TrainChPoint['categories']=categories
        TrainChPoint['class_to_idx']=class_to_idx
        TrainChPoint['idx_to_class']=idx_to_class
        torch.save(TrainChPoint, save_file_name) 

        # # Training Results
        # We can inspect the training progress by looking at the `history`. 
        # plot loss
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'val_loss', 'test_loss']:
            plt.plot(
                history[c], label=c) 
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(save_path+f'/LossPerEpoch_fold_{fold_idx}.png')
        # plt.show()
        # plot accuracy
        plt.figure(figsize=(8, 6))
        for c in ['train_acc', 'val_acc', 'test_acc']:
            plt.plot(
                100 * history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch') 
        plt.ylabel('Accuracy') 
        plt.savefig(save_path+f'/AccuracyPerEpoch_fold_{fold_idx}.png')
        # plt.show()

        # # Create test labels
        # _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)   
        # labels_test = torch.from_numpy(labels_test).to(torch.int64) 
        # # test dataloader
        # test_ds = MyData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std)
        # test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)

        # release memeory (delete variables)
        del  my_transforms, optimizer, scheduler
        del  train_ds, train_dl, val_ds, val_dl
        del  img_names_test, labels_test 
        del  TrainChPoint
        torch.cuda.empty_cache()

        # # Test Accuracy
        all_paths =list()
        test_acc = 0.0
        test_loss = 0.0
        i=0
        model.eval() 
        for data, targets, im_path in test_dl:

            # Tensors to gpu
            if train_on_gpu:
                data = data.to('cuda', non_blocking=True)
                targets = targets.to('cuda', non_blocking=True)
            
            # all_targets = torch.cat([all_targets ,targets.numpy()])
            # Raw model output
            out = model(data)
            loss = criterion(out, targets)
            test_loss += loss.item() * data.size(0)
            out = torch.exp(out)
            # pred_probs = torch.cat([pred_probs ,out])
            all_paths.extend(im_path)
            targets = targets.cpu()
            if i==0:
                all_targets = targets.numpy()
                pred_probs = out.cpu().detach().numpy()
            else:
                all_targets = np.concatenate((all_targets  ,targets.numpy()))
                pred_probs = np.concatenate((pred_probs  ,out.cpu().detach().numpy()))
            _, temp_label = torch.max(out.cpu(), dim=1)
            correct_tensor = temp_label.eq(targets.data.view_as(temp_label))      # this lin is temporary 
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))   # this lin is temporary 
            test_acc += accuracy.item() * data.size(0)                      # this lin is temporary 
            temp_label = temp_label.detach().numpy()
            if i==0:
                pred_label = temp_label
            else:
                pred_label = np.concatenate((pred_label  ,temp_label))

            i +=1
        test_loss = test_loss / len(test_dl.dataset)
        test_loss = round(test_loss,4)
        test_acc = test_acc / len(test_dl.dataset)                          # this lin is temporary
        test_acc = round(test_acc*100,2)
        print(f'Test Loss: {test_loss},  Test Accuracy: {test_acc}%')


        from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
        # main confusion matrix
        cm = confusion_matrix(all_targets, pred_label)
        # it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
        # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
        cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)

        # # Saving Test Results
        save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = {} 
        TestChPoint['categories']=categories
        TestChPoint['class_to_idx']=class_to_idx
        TestChPoint['idx_to_class']=idx_to_class
        TestChPoint['Train_history']=history 
        TestChPoint['n_Class_test']=n_Class_test
        TestChPoint['targets']=all_targets
        TestChPoint['prediction_label']=pred_label
        TestChPoint['prediction_probs']=pred_probs
        TestChPoint['image_names']=all_paths 
        TestChPoint['cm']=cm
        TestChPoint['cm_per_class']=cm_per_class
        torch.save(TestChPoint, save_file_name)
        # torch.load(save_file_name) 
       
        # release memeory (delete variables)
        del model, criterion, history, test_ds, test_dl
        del data, targets, out, temp_label, 
        del test_acc, test_loss, loss
        del pred_probs, pred_label, all_targets, all_paths, 
        del cm, cm_per_class, TestChPoint
        torch.cuda.empty_cache()
        print(f'completed fold {fold_idx}')

    print('#############################################################')

    # delete checkpoint 
    os.remove(checkpoint_name)
    print("Checkpoint File Removed!")

    # Overall Test results
    load_path = Results_path +'/'+ model_name
    for fold_idx in range(loop_start, loop_end):
        fold_path = load_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = torch.load(fold_path)
        if fold_idx==loop_start:
            cumulative_cm = TestChPoint['cm']
        else:
            cumulative_cm += TestChPoint['cm']
    Overall_Accuracy = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm)
    Overall_Accuracy = round(Overall_Accuracy*100, 2)
    print('Cummulative Confusion Matrix')
    print(cumulative_cm)
    print(f'Overall Test Accuracy: {Overall_Accuracy}')

    print('#############################################################')



