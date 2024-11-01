# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import pandas as pd
import os
from os import path
from importlib import import_module
from scipy.io import loadmat, savemat
# Image manipulations
from PIL import Image
# customized functions 
from utils import *
from models import *
# Timing utility
from timeit import default_timer as timer
from selfonn import SelfONNLayer

 
# Parse command line arguments
fname = "config_inference.py" 
configuration = import_module(fname.split(".")[0])
config = configuration.config

if __name__ ==  '__main__':  
    # torch.set_num_threads(1)
    ################## Network hyper-parameters 
    parentdir = config['parentdir']                     # main directory
    input_type  = config['input_type']                  # 'NIFTI' or 'video'
    ONN = config['ONN']                                 # set to 'True' if you are using ONN
    input_ch = config['input_ch']                       # 1 for gray scale x-rays, and 3 for RGB (3channel) x-rays  
    batch_size = config['batch_size']                   # batch size, Change to fit hardware
    num_folds = config['num_folds']                     # number of cross validation folds
    CI = config['CI']                                   # Confidence interval (missied cases with probability>=CI will be reported in excel file)
    input_mean = config['input_mean']                   # dataset mean per channel
    input_std = config['input_std']                     # dataset std per channel
    Resize_h = config['Resize_h']                       # network input size
    Resize_w = config['Resize_w']  
    load_model = config['load_model']                   # specify full path of pretrained model pt file or set to False to load trained model
    labeled_Data =  config['labeled_Data']              # set to true if you have the labeled test set
    model_name = config['model_name']                   # name of trained model .pt file
    new_name = config['new_name']                       # specify a new folder name to save test results, else set to False to overwrite test results genertaed by train code
    Results_path = config['Results_path']               # main results file
    save_path = config['save_path']                     # save path 
    fold_to_run = config['fold_to_run']                 # define as [] to loop through all folds, or specify start and end folds i.e. [3 5]
    ################## 
    # test Directory 
    testdir = parentdir + 'Data/Test/' 
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
        multi_gpu = False

    test_history = []
    index = []
    # loop through folds
    if not fold_to_run:
        loop_start = 1
        loop_end = num_folds+1
    else:
        loop_start = fold_to_run[0]
        loop_end = fold_to_run[1]+1
    for fold_idx in range(loop_start, loop_end):
        print('#############################################################')
        print(f'started fold {fold_idx}')
        # save_file_name = save_path + '/' + model_name  + f'_fold_{fold_idx}.pt'
        testdir_fold = testdir + f'fold_{fold_idx}/' 

        
        # load model 
        if load_model:
            checkpoint = torch.load(load_model)
            model = checkpoint['model'] 
            if len(checkpoint)>=5:  # if length is less than 5, then file was created by the old code
                categories = checkpoint['categories']
                class_to_idx = checkpoint['class_to_idx']
                idx_to_class = checkpoint['idx_to_class'] 
            del checkpoint 
        else: 
            pt_file = Results_path+ '/' + model_name + '/' + model_name + f'_fold_{fold_idx}.pt'
            checkpoint = torch.load(pt_file)
            model = checkpoint['model'] 
            # # temp 
            # # history = checkpoint['history']
            # del checkpoint
            # categories = ['Normal', 'COVID', 'non_COVID'] 
            # class_to_idx = {'Normal':0, 'COVID':1, 'non_COVID':2}
            # idx_to_class = {0:'Normal', 1:'COVID', 2:'non_COVID'} 
            # checkpoint = {}
            # # checkpoint['history'] = history
            # checkpoint['model'] = model
            # checkpoint['categories'] = categories
            # checkpoint['class_to_idx'] = class_to_idx
            # checkpoint['idx_to_class'] = idx_to_class
            # torch.save(checkpoint,pt_file) 
            # # temp
            if len(checkpoint)>=5:  # if length is less than 5, then file was created by the old code
                categories = checkpoint['categories']
                class_to_idx = checkpoint['class_to_idx']
                idx_to_class = checkpoint['idx_to_class'] 
            del   pt_file, checkpoint
        model.eval()
        model = model.to('cuda')  
 
        # check if model on cuda
        if next(model.parameters()).is_cuda:
            print('model device: cuda')
    
        # choose model loss function 
        criterion = nn.NLLLoss() 

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
                ])   

                
        # test dataloader
        # Create test labels
        if class_to_idx: # if class_to_idx is not defined, then file was created by the old code 
            n_Class_test, img_names_test, labels_test = Retrievelabel(testdir_fold,categories) 
            labels_test = torch.from_numpy(labels_test).to(torch.int64) 
        else:
            categories, n_Class_test, img_names_test, labels_test, class_to_idx, idx_to_class = Createlabels(testdir_fold)    
        # test dataloader
        if input_type=='NIFTI': 
            test_ds = MyData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std)
        elif input_type=='video':
            test_ds = MyVideosData(root_dir=testdir_fold,categories=categories,img_names=img_names_test,target=labels_test,my_transforms=my_transforms,return_path=True,ONN=ONN,mean=input_mean,std=input_std,width=Resize_w,height=Resize_h) 
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1)
        test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,pin_memory=True,num_workers=1) 

        if labeled_Data:
            all_paths =list()
            test_acc = 0.0
            test_loss = 0.0
            i=0
            for data, targets, im_path in test_dl:
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)
                # model output
                # input_time = timer()
                out = model(data)
                # output_time = timer() - input_time
                # print(output_time) 
                # loss
                loss = criterion(out, targets)
                test_loss += loss.item() * data.size(0)
                out = torch.exp(out)
                # images names
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
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))         # this lin is temporary 
                test_acc += accuracy.item() * data.size(0)                            # this lin is temporary 
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
            #
            from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
            # main confusion matrix
            cm = confusion_matrix(all_targets, pred_label)
            # it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
            # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
            cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)
            # # Saving Test Results
            if new_name:
                save_file_name = save_path + '/' + new_name + f'_test_fold_{fold_idx}.pt'
            else:
                save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test
            TestChPoint['targets']=all_targets
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['image_names']=all_paths 
            TestChPoint['cm']=cm
            TestChPoint['cm_per_class']=cm_per_class
            torch.save(TestChPoint, save_file_name)
        else:
            all_paths =list()
            i=0
            for data, _, im_path in test_dl: 
                # Tensors to gpu
                if train_on_gpu:
                    data = data.to('cuda', non_blocking=True)
                # model output
                out = model(data)
                out = torch.exp(out)
                # images names
                all_paths.extend(im_path)
                if i==0:
                    pred_probs = out.cpu().detach().numpy()
                else:
                    pred_probs = np.concatenate((pred_probs  ,out.cpu().detach().numpy()))
                _, temp_label = torch.max(out.cpu(), dim=1)
                temp_label = temp_label.detach().numpy()
                if i==0:
                    pred_label = temp_label
                else:
                    pred_label = np.concatenate((pred_label  ,temp_label))
                i +=1
            # # Saving Test Results
            if new_name:
                save_file_name = save_path + '/' + new_name + f'_test_fold_{fold_idx}.pt'
            else:
                save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'

            TestChPoint = {} 
            TestChPoint['categories']=categories
            TestChPoint['class_to_idx']=class_to_idx
            TestChPoint['idx_to_class']=idx_to_class
            TestChPoint['n_Class_test']=n_Class_test 
            TestChPoint['prediction_label']=pred_label
            TestChPoint['prediction_probs']=pred_probs
            TestChPoint['image_names']=all_paths 
            torch.save(TestChPoint, save_file_name) 

       # release memeory (delete variables)
        if labeled_Data: 
            del model, criterion, test_ds, test_dl
            del data, targets, out, temp_label  
            del test_acc, test_loss, loss
            del pred_probs, pred_label, all_targets, all_paths, 
            del cm, cm_per_class, TestChPoint
        else:
            del model, criterion, test_ds, test_dl
            del data, out, temp_label  
            del pred_probs, pred_label, all_paths, 
            del TestChPoint 
        torch.cuda.empty_cache()
        print(f'completed fold {fold_idx}') 
    
    print('#############################################################') 


    if labeled_Data: 
        all_Missed_c =list() 
        for fold_idx in range(loop_start, loop_end):
            # load checkpoint
            if new_name:
                fold_path = save_path + '/' + new_name + f'_test_fold_{fold_idx}.pt'
            else:
                fold_path = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(fold_path)
            if fold_idx==loop_start:
                targets = TestChPoint['targets']
                pred = TestChPoint['prediction_label']
                pred_probs = TestChPoint['prediction_probs']
                image_names  = TestChPoint['image_names']
            else: 
                targets = np.concatenate([targets, TestChPoint['targets']])
                pred = np.concatenate([pred, TestChPoint['prediction_label']])
                pred_probs = np.concatenate([pred_probs, TestChPoint['prediction_probs']])
                image_names.extend(TestChPoint['image_names'])
            # find missed cases (probs, image path)
            n = len(TestChPoint['categories'])
            # temp
            current_fold_target = TestChPoint['targets']
            current_fold_pred = TestChPoint['prediction_label']
            current_fold_image_names = TestChPoint['image_names'] 
            current_fold_prediction_probs = TestChPoint['prediction_probs'] 
            # missed_idx = np.argwhere(1*(targets==pred) == 0)
            missed_idx = np.argwhere(1*(current_fold_target==current_fold_pred) == 0)
            # temp 
            m = len(missed_idx)
            missed_probs = np.zeros((m,n)) 
            for i in range(len(missed_idx)):
                index = int(missed_idx [i])
                all_Missed_c.extend([ f'fold_{fold_idx}/'+current_fold_image_names[index] ])
                missed_probs[i,:] = current_fold_prediction_probs[index,:] 
            if fold_idx==loop_start:
                all_missed_p = missed_probs
            else: 
                all_missed_p = np.concatenate((all_missed_p,missed_probs))
        # find missed cases with high CI (probs, image path)
        temp = np.max(all_missed_p,axis=1)
        temp_idx = np.argwhere(temp >= CI) 
        unsure_missed_c = list() 
        unsure_missed_p = np.zeros((len(temp_idx), n)) 
        for i in range(len(temp_idx)):
            index = int(temp_idx[i])
            unsure_missed_c.extend([ all_Missed_c[index] ]) 
            unsure_missed_p[i,:] =  all_missed_p[index,:]

        categories = TestChPoint['categories']
        n = len(categories)
        from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
        # main confusion matrix
        cm = confusion_matrix(targets, pred)
        # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
        # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP   
        cm_per_class = multilabel_confusion_matrix(targets, pred)
        # Overall Accuracy
        Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
        Overall_Accuracy = round(Overall_Accuracy*100, 2)

        # create missed and unsure missed tables
        missed_table = pd.DataFrame(all_Missed_c, columns=[f'Missed Cases'])
        unsure_table = pd.DataFrame(unsure_missed_c, columns=[f'Unsure Missed Cases (CI={CI})']) 
        missed_prob_table = pd.DataFrame(np.round(all_missed_p,4), columns=categories) 
        unsure_prob_table = pd.DataFrame(np.round(unsure_missed_p,4), columns=categories) 


        # create confusion matrix table (pd.DataFrame)
        cm_table = pd.DataFrame(cm, index=TestChPoint['categories'] , columns=TestChPoint['categories'])

        Eval_Mat = []
        # per class metricies
        for i in range(len(categories)):
            TN = cm_per_class[i][0][0] 
            FP = cm_per_class[i][0][1]   
            FN = cm_per_class[i][1][0]  
            TP = cm_per_class[i][1][1]  
            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
            Precision = round(100*(TP)/(TP+FP), 2)  
            Sensitivity = round(100*(TP)/(TP+FN), 2) 
            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
            Specificity = round(100*(TN)/(TN+FP), 2)  
            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        # sizes of each class
        s = np.sum(cm,axis=1) 
        # create tmep excel table 
        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
        # weighted average of per class metricies
        Accuracy = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
        Precision = round(temp_table['Precision'].dot(s)/np.sum(s), 2)  
        Sensitivity = round(temp_table['Sensitivity'].dot(s)/np.sum(s), 2)  
        F1_score = round(temp_table['F1_score'].dot(s)/np.sum(s), 2)  
        Specificity = round(temp_table['Specificity'].dot(s)/np.sum(s), 2)   
        values = [Accuracy, Precision, Sensitivity, F1_score, Specificity]
        # create per class metricies excel table with weighted average row
        Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
        categories.extend(['Weighted Average'])
        Eval_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)

        # create confusion matrix table (pd.DataFrame)
        Overall_Acc = pd.DataFrame(Overall_Accuracy, index=['Overall_Accuracy'] , columns=[' '])

        # print('Cummulative Confusion Matrix')
        print('\n') 
        print(cm_table) 
        print('\n') 
        # print('Evaluation Matricies')
        print(Eval_table)
        print('\n')  
        # print('Overall Accuracy')
        print(Overall_Acc)
        print('\n') 
    
        # save to excel file   
        if new_name:
            new_savepath = save_path +'/'+  new_name +'.xlsx'  # file to save 
        else: 
            new_savepath = save_path +'/'+  model_name +'.xlsx'  # file to save 
        writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
        # sheet 1 (Unsure missed cases) + (Evaluation metricies) + (Commulative Confusion Matrix) 
        col =0; row =2 
        unsure_table.to_excel(writer, "Results", startcol=col,startrow=row) 
        col =2; row =2 
        unsure_prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
        col =col+n+2; row =2 
        Eval_table.to_excel(writer, "Results", startcol=col,startrow=row)
        row = row +len(class_to_idx)+7
        Overall_Acc.to_excel(writer, "Results", startcol=col,startrow=row, header=None)
        col = col+8; row=1   
        Predicted_Class = pd.DataFrame(['Predicted Class'])
        Predicted_Class.to_excel(writer, "Results", startcol=col+1,startrow=row, header=None, index=None)
        row =2     
        cm_table.to_excel(writer, "Results", startcol=col,startrow=row)
        # sheet 2 (All missed cases)
        col =0; row =2 
        missed_table.to_excel(writer, "Extra", startcol=col,startrow=row)
        col =2; row =2 
        missed_prob_table.to_excel(writer, "Extra", startcol=col,startrow=row, index=None)
        # save 
        writer.save()  
        # new 
        # Save needed variables to create ROC curves 
        ROC_checkpoint = {} 
        ROC_checkpoint['prediction_label'] = pred
        ROC_checkpoint['prediction_probs'] = pred_probs
        ROC_checkpoint['targets'] = targets
        ROC_checkpoint['class_to_idx']=class_to_idx
        ROC_checkpoint['idx_to_class']=idx_to_class 
        if new_name:
            ROC_path_pt = save_path +'/'+  new_name +'_roc_inputs.pt'  # file to save 
            ROC_path_mat = save_path +'/'+  new_name +'_roc_inputs.mat'  # file to save 
        else: 
            ROC_path_pt = save_path +'/'+  model_name +'_roc_inputs.pt'  # file to save 
            ROC_path_mat = save_path +'/'+  model_name +'_roc_inputs.mat'  # file to save 
        torch.save(ROC_checkpoint,ROC_path_pt) 
        # import scipy.io as spio 
        # spio.savemat(ROC_path_mat, ROC_checkpoint) 
        savemat(ROC_path_mat, ROC_checkpoint) 
        # new
    else:
        for fold_idx in range(loop_start, loop_end):
            # load checkpoint 
            if new_name:
                fold_path = save_path + '/' + new_name + f'_test_fold_{fold_idx}.pt'
            else:
                fold_path = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
            TestChPoint = torch.load(fold_path)
            #
            categories = TestChPoint['categories']  
            #
            temp_pred = TestChPoint['prediction_label'] 
            pred_probs = TestChPoint['prediction_probs'] 
            image_names  = TestChPoint['image_names']
            for i in range(len(temp_pred)): 
                if i==0:
                    pred = [ idx_to_class[temp_pred[i]] ] 
                else:
                    pred.extend([ idx_to_class[temp_pred[i]] ])  
            # create missed and unsure missed tables
            input_names_table = pd.DataFrame(image_names, columns=[f'Input Image']) 
            pred_table = pd.DataFrame(pred, columns=[f'Prediction']) 
            prob_table = pd.DataFrame(np.round(pred_probs,4), columns=categories) 
            # save to excel file   
            if new_name:
                new_savepath = save_path +'/'+  new_name + f'_fold{fold_idx}.xlsx'  # file to save 
            else: 
                new_savepath = save_path +'/'+  model_name +'.xlsx'  # file to save 
            writer = pd.ExcelWriter(new_savepath, engine='openpyxl')
            # sheet 1 (input images) + (predictions) + (predictions probabilities) 
            col =0; row =2 
            input_names_table.to_excel(writer, "Results", startcol=col,startrow=row)
            col =2; row =2  
            pred_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            col =3; row =2 
            prob_table.to_excel(writer, "Results", startcol=col,startrow=row, index=None)
            # save 
            writer.save()  


    print('#############################################################') 





 







    