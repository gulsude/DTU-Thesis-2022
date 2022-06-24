import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy
import seaborn as sn

from IPython.display import HTML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc

import gc
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

sys.path.append('/home/s202357/thesis/transmut/pipeline/procedure/architecture')
from model_components import MyDataSet, color

device = torch.device('cuda')
criterion = nn.CrossEntropyLoss()
data_dir = '/home/s202357/thesis/transmut/data/transmut_github/'
vocab = np.load( data_dir + 'Transformer_vocab_dict.npy', allow_pickle = True).item()
pep_max_len = 14
hla_max_len = 34


blosum_dict = {
'A' : [5,-2,-1,-2,-1,-1,-1,0,-2,-1,-2,-1,-1,-3,-1,1,0,-3,-2,0],
'R' : [-2,7,-1,-2,-4,1,0,-3,0,-4,-3,3,-2,-3,-3,-1,-1,-3,-1,-3],
'N' : [-1,-1,7,2,-2,0,0,0,1,-3,-4,0,-2,-4,-2,1,0,-4,-2,-3],
'D' : [-2,-2,2,8,-4,0,2,-1,-1,-4,-4,-1,-4,-5,-1,0,-1,-5,-3,-4],
'C' : [-1,-4,-2,-4,13,-3,-3,-3,-3,-2,-2,-3,-2,-2,-4,-1,-1,-5,-3,-1],
'Q' : [-1,1,0,0,-3,7,2,-2,1,-3,-2,2,0,-4,-1,0,-1,-1,-1,-3],
'E' : [-1,0,0,2,-3,2,6,-3,0,-4,-3,1,-2,-3,-1,-1,-1,-3,-2,-3],
'G' : [0,-3,0,-1,-3,-2,-3,8,-2,-4,-4,-2,-3,-4,-2,0,-2,-3,-3,-4],
'H' : [-2,0,1,-1,-3,1,0,-2,10,-4,-3,0,-1,-1,-2,-1,-2,-3,2,-4],
'I' : [-1,-4,-3,-4,-2,-3,-4,-4,-4,5,2,-3,2,0,-3,-3,-1,-3,-1,4],
'L' : [-2,-3,-4,-4,-2,-2,-3,-4,-3,2,5,-3,3,1,-4,-3,-1,-2,-1,1],
'K' : [-1,3,0,-1,-3,2,1,-2,0,-3,-3,6,-2,-4,-1,0,-1,-3,-2,-3],
'M' : [-1,-2,-2,-4,-2,0,-2,-3,-1,2,3,-2,7,0,-3,-2,-1,-1,0,1],
'F' : [-3,-3,-4,-5,-2,-4,-3,-4,-1,0,1,-4,0,8,-4,-3,-2,1,4,-1],
'P' : [-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,-1,-1,-4,-3,-3],
'S' : [1,-1,1,0,-1,0,-1,0,-1,-3,-3,0,-2,-3,-1,5,2,-4,-2,-2],
'T' : [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,2,5,-3,-2,0],
'W' : [-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1,1,-4,-4,-3,15,2,-3],
'Y' : [-2,-1,-2,-3,-3,-1,-2,-3,2,-1,-1,-2,0,4,-3,-2,-2,2,8,-1],
'V' : [0,-3,-3,-4,-1,-3,-3,-4,-4,4,1,-3,1,-1,-3,-2,0,-3,-1,5]
}


def pep_to_blosum(pep):
    return [blosum_dict[word] for word in pep]


def make_data_bl(data, pep_max_len, hla_max_len):
        pep_inputs, hla_inputs, labels = [], [], []
        cc = 0
        tt = time.time()
        for pep, hla, label in zip(data.peptide, data.HLA_sequence, data.label):
            bl = pep_to_blosum(pep)
            pep_len = len(pep)
            for t in range(pep_max_len-pep_len):
                xs = [-1]*20
                bl.append(xs)

            pep_inputs.append(bl)
            bl_hla = pep_to_blosum(hla) 
            hla_inputs.append(bl_hla)

            cc += 1

        return torch.tensor(pep_inputs, dtype=torch.float).to(device), torch.tensor(hla_inputs, dtype=torch.float).to(device)

    
def make_data_emb(data, pep_max_len, hla_max_len, vocab):
        pep_inputs, hla_inputs, labels = [], [], []
        for pep, hla, label in zip(data.peptide, data.HLA_sequence, data.label):
            pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
            pep_input = [[vocab[n] for n in pep]] 
            hla_input = [[vocab[n] for n in hla]]
            pep_inputs.extend(pep_input)
            hla_inputs.extend(hla_input)
            labels.append(label)
        return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)
    


def get_attn_pad_mask_fake(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    
    batch_size, len_q, _ = seq_q.size()
    batch_size, len_k, _ = seq_k.size()
    
    seq_q = torch.zeros(batch_size, len_q).to(device)
    seq_k = torch.zeros(batch_size, len_k).to(device)
    
    pad_attn_mask = seq_k.data.eq(1).unsqueeze(1) 
        
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    
    
def best_treshold(df_results, t= []):
    acc_best = -999
    best_i = -999

    if t == []:
        for i in np.arange(0, 1, 0.1):
            df_results['pred_binary'] = np.where((df_results.pred > i) , 1, 0)
            acc = accuracy_score(df_results['target'], df_results['pred_binary'])
            if acc > acc_best:
                df = df_results
                best_i = i
                acc_best = acc
    else:
        best_i = t[0]
        df_results['pred_binary'] = np.where((df_results.pred > t[0]) , 1, 0)
        df = df_results
    return acc_best, best_i, df


def pkl(id_, n_layers, n_heads, model_num, d_model):
    return '{}_d{}_layer{}_multihead{}_MODEL{}.pkl'.format(id_, d_model, n_layers, n_heads, model_num)

    
def performances(y_true, y_pred, y_prob, print_ = True):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    try:
        mcc = ((tp*tn) - (fn*fp)) / np.sqrt(np.float((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    except:
        print('MCC Error: ', (tp+fn)*(tn+fp)*(tp+fp)*(tn+fn))
        mcc = np.nan
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    try:
        recall = tp / (tp+fn)
    except:
        recall = np.nan
        
    try:
        precision = tp / (tp+fp)
    except:
        precision = np.nan
        
    try: 
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = np.nan
        
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))
    
    return (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr)


def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


f_mean = lambda l: sum(l)/len(l)


def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', 'precision', 'recall', 'aupr']

    performances_pd = pd.DataFrame(performances_list, columns = metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis = 0)
    performances_pd.loc['std'] = performances_pd.std(axis = 0)
    
    return performances_pd


def binary_roc_auc_score(y_true, y_score, sample_weight=None, max_fpr=None):
    """Binary roc auc score."""
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. ROC AUC score "
            "is not defined in that case.")

    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected max_fpr in range (0, 1], got: %r" % max_fpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)
    return partial_auc/0.1


def make_data_padding(data_peptide):
        pep_inputs_to_pad = []
        for pep in data_peptide:
            pep = pep.ljust(pep_max_len, '-')
            pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
            pep_inputs_to_pad.extend(pep_input)
        return torch.LongTensor(pep_inputs_to_pad)

    
def data_with_loader_unique(data_dir, model_type, pep_max_len, hla_max_len, vocab, index_order_idx, type_ = 'train', fold = None,  batch_size = 1024):
        
    pep_inputs = []
    hla_inputs = []
    labels = []
    
    if 'MHC' in model_type:
        data = pd.read_csv(data_dir + 'upd_{}_mhc.csv'.format(fold), index_col = 0).drop_duplicates()
    else:
        data = pd.read_csv(data_dir + 'upd_{}_d.csv'.format(fold), index_col = 0).drop_duplicates()
    
    if "blosum" in model_type.lower():
        pep_inputs, hla_inputs = make_data_bl(data, pep_max_len, hla_max_len)
        pep_to_pad = make_data_padding(data.peptide)
        labels = torch.LongTensor( [x for x in data.label] ).to(device)
        loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, pep_to_pad, labels), batch_size, shuffle = False, num_workers = 0)    
    else:
        pep_inputs, hla_inputs, labels = make_data_emb(data, pep_max_len, hla_max_len, vocab)
        loader = Data.DataLoader(MyDataSet(pep_inputs, hla_inputs, pep_inputs, labels), batch_size, shuffle = False, num_workers = 0)
  
    print("\tTest File ID", fold, len(pep_inputs), np.shape(pep_inputs))

    return data, pep_inputs, hla_inputs, labels, loader

def train_step(model, train_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    tr = 0
    for train_pep_inputs, train_hla_inputs, train_labels in train_loader:
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        train_outputs: [batch_size, 2]
        '''
        tr += 1
        train_pep_inputs, train_hla_inputs, train_labels = train_pep_inputs.to(device), train_hla_inputs.to(device), train_labels.to(device)

        t1 = time.time()
        train_outputs, _, _ = model(train_pep_inputs, train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim = 1)(train_outputs)[:, 1].cpu().detach().numpy()
        
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)
#         dec_attns_train_list.append(train_dec_self_attns)
        
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    
    print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_ = True)
    
    return ys_train, loss_train_list, metrics_train, time_train_ep#, dec_attns_train_list


def eval_step(model, model_type, threshold, val_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device('cuda')
    
    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        for val_pep_inputs, val_hla_inputs, val_labels in val_loader:
            val_pep_inputs, val_hla_inputs, val_labels = val_pep_inputs.to(device), val_hla_inputs.to(device), val_labels.to(device)
            if model_type == 'ED':
                val_outputs, _, _, _ = model(val_pep_inputs, val_hla_inputs)
            else:
                val_outputs, _, _ = model(val_pep_inputs, val_hla_inputs)
            val_loss = criterion(val_outputs, val_labels)

            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
#             dec_attns_val_list.append(val_dec_self_attns)
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print('Fold-{} ****Test  Epoch-{}/{}: Loss = {:.6f}'.format(fold, epoch, epochs, f_mean(loss_val_list)))
        
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list)
    return ys_val, loss_val_list, metrics_val, y_prob_val_list, y_pred_val_list #, dec_attns_val_list


def eval_step_test(model, model_type, threshold, val_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device('cuda')
    
    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list, attns_val_list, attns_hla_val_list = [], [], [], []
        for val_pep_inputs, val_hla_inputs, _, val_labels in val_loader:
            val_pep_inputs, val_hla_inputs, val_labels = val_pep_inputs.to(device), val_hla_inputs.to(device), val_labels.to(device)
            
            if model_type == 'ED':
                val_outputs, enc_attn, enc_hla_attn, _ = model(val_pep_inputs, val_hla_inputs)
            else:
                val_outputs, enc_attn, enc_hla_attn = model(val_pep_inputs, val_hla_inputs, val_pep_inputs)
            val_loss = criterion(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
            
            if model_type != 'FFNN':
                attns_val_list.extend(enc_attn[0].cpu().numpy())
                attns_hla_val_list.extend(enc_hla_attn[0].cpu().numpy())
            
        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
        
        print_bool = False
        if epoch == 1 or epoch == 25 or epoch == 50:
            print_bool = True
            print('\nFold-{} **** Test  Epoch-{}/{}: Loss = {:.6f} | '.format(fold, epoch, epochs, f_mean(loss_val_list)), end='')
            
        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = print_bool)
    return ys_val, loss_val_list, metrics_val, y_prob_val_list, y_pred_val_list, attns_val_list,  attns_hla_val_list#, dec_attns_val_list

def sort_aatype(df):
    aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
    df.reset_index(inplace = True)
    df['index'] = df['index'].astype('category')
    df['index'].cat.reorder_categories(aatype_sorts, inplace=True)
    df.sort_values('index', inplace=True)
    df.rename(columns = {'index':''}, inplace = True)
    df = df.set_index('')
    return df


def eval_step_test_bl(model, model_abbr, threshold, val_loader, fold, epoch, epochs, use_cuda = True):
    device = torch.device('cuda')

    model.eval()
    torch.manual_seed(19961231)
    torch.cuda.manual_seed(19961231)
    with torch.no_grad():
        loss_val_list, dec_attns_val_list = [], []
        y_true_val_list, y_prob_val_list = [], []
        attns_val_list, attns_hla_val_list = [], []
        for val_pep_inputs, val_hla_inputs, pep_to_pad, val_labels in val_loader:
            val_pep_inputs, val_hla_inputs, pep_to_pad, val_labels = val_pep_inputs.to(device), val_hla_inputs.to(device), pep_to_pad.to(device), val_labels.to(device)
            val_outputs, enc_attn, enc_hla_attn = model(val_pep_inputs, val_hla_inputs, pep_to_pad)
            val_loss = criterion(val_outputs, val_labels)
            y_true_val = val_labels.cpu().numpy()
            y_prob_val = nn.Softmax(dim = 1)(val_outputs)[:, 1].cpu().detach().numpy()

            attns_val_list.extend(enc_attn[0].cpu().numpy())
            attns_hla_val_list.extend(enc_hla_attn[0].cpu().numpy())
            
            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)
            loss_val_list.append(val_loss)
#             dec_attns_val_list.append(val_dec_self_attns)

        y_pred_val_list = transfer(y_prob_val_list, threshold)
        ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

        print_bool = False
        if epoch == 1 or epoch == 25 or epoch == 50:
            print_bool = True
            print('\nFold-{} **** Test  Epoch-{}/{}: Loss = {:.6f} | '.format(fold, epoch, epochs, f_mean(loss_val_list)), end='')

        metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = print_bool)
    return ys_val, loss_val_list, metrics_val, y_prob_val_list, y_pred_val_list, attns_val_list, attns_hla_val_list
         
'''    
def binary_pseudo_seq(pseudo, mhc_seq, max_list):
    bin_, idx_list, idx_pos, idx_val, idx_val_bin = [], [], [], [], []
    for p in pseudo:
        for m in range(len(mhc_seq)):
            if m in idx_list:
                continue
            else:
                if p != mhc_seq[m]:
                    idx_list.append(m)
                    bin_.append(0)
                else:
                    idx_val.append(max_list[m])
                    idx_val_bin.append(1)
                    idx_list.append(m)
                    idx_pos.append(m)
                    bin_.append(1)
                    break  
    return bin_
''' 

def print_seq(aa, value, threshold):
    if value > threshold:
        print('' + color.BOLD + color.RED + aa + color.BOLD, end='')
        return aa
    else:
        print(color.BOLD + color.CYAN + aa, end="")
        return None
            
            
def sort_aatype(df):
    aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
    df.reset_index(inplace = True)
    df['index'] = df['index'].astype('category')
    df['index'].cat.reorder_categories(aatype_sorts, inplace=True)
    df.sort_values('index', inplace=True)
    df.rename(columns = {'index':''}, inplace = True)
    df = df.set_index('')
    return df

def attn_HLA_length_aatype_position_num_HLApseudo(data, attn_data, n_heads, hla, label = None, length = 9, show_num = False):
    aatype_position = dict()
    if label == None:
        if length == None:
            length_index = np.array(data[data.HLA == hla][data.label == label].index)
        else:
            length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)
    elif length == None:
            length_index = np.array(data[data.HLA == hla][data.label == label].index)
    else:
        length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)

        print("\n\nHLA and Label are filtered.")

    length_data_num = len(length_index)
    print("#Peptides:", length_data_num)
    print("HLA:", hla)
    for head in range(n_heads):
        cc = -1
        p = False
        for idx in length_index:
            p = False
            cc += 1
            temp_hla = data.iloc[idx].HLA_sequence
            temp_length_head_ = attn_data[idx][head]
            temp_length_head = temp_length_head_.sum(axis = 0)
            for i, aa in enumerate(temp_hla): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 
    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_hla = data.iloc[idx].HLA_sequence
            for i, aa in enumerate(temp_hla):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, length_data_num, aatype_position_num
    else:
        return aatype_position, length_data_num
    
    
def draw_hla_length_aatype_position_HLApseudo(s1, s2, cl, data, attn_data, n_heads, hla, hla_size, softmax_dim = -1, label = None, length = 9, threshold = 0.95, top = True, show = True, softmax = True, unsoftmax = True):
    
    func = lambda x: round(x,2)
    HLA_length_aatype_position, length_data_num = attn_HLA_length_aatype_position_num_HLApseudo(data, attn_data, n_heads, hla, label, length, show_num = False)
    
    if softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd_HLApseudo(
                                                                                     HLA_length_aatype_position, 
                                                                                     length_data_num,
                                                                                     softmax_dim,
                                                                                     length,
                                                                                     hla_size,
                                                                                     softmax,
                                                                                     unsoftmax)

        HLA_length_aatype_position_softmax_pd = sort_aatype(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_unsoftmax_pd = sort_aatype(HLA_length_aatype_position_unsoftmax_pd)
        
        max_list = []
        hla_len = 0
        for (columnName, columnData) in HLA_length_aatype_position_softmax_pd.iteritems():
            hla_len += 1
            value = np.max(columnData.values, axis=0)
            max_list.append(value)
            
        max_list = list(map(func, max_list))
        
        threshold_inner = round(np.max(np.array(max_list)) * threshold, 2)
        print("Threshold:", threshold_inner)
        new_pseudo = []
        new_pseudo_pos = []
        new_pseudo_val = []
        
        pos = 0
        mhc_seq = ''
        mhc_seq_pos = []
        for (columnName, columnData) in HLA_length_aatype_position_softmax_pd.iteritems():
            aa = list(blosum_dict.keys())[np.argmax(columnData.values, axis=0)][0]
            mhc_seq = mhc_seq + aa
            mhc_seq_pos.append(str(pos+1)+'\n'+aa)
            value = np.max(columnData.values, axis=0)
            aa = print_seq(aa, value, threshold_inner)
            if aa:
                new_pseudo.append(aa)
                new_pseudo_pos.append(pos)
                new_pseudo_val.append(value)
            pos += 1
        
        if show == 'bar':
            if top:
                threshold_inner = np.array(new_pseudo_val).min()
            print(color.END + "\nthreshold:", threshold_inner)
            
            sn.set(rc = {'figure.figsize':(s1,s2)})
            sn.set_style("white")
            ax = sn.barplot(x=np.arange(len(max_list)), y=max_list)
            ax.set_xticklabels(mhc_seq)
            ax.set_yticklabels([round(x,2) for x in ax.get_yticks()], size=22)
            ax.set_xticklabels(mhc_seq_pos, size=22)
            #ax.axhline(threshold_inner)
            #plt.legend(loc='upper right', labels=['amino acid in MHC pseudo sequence'])
            idx = 0

            for bar in ax.patches:
                bar.set_color(cl)     
            
            plt.show()
        if show == 'heatmap':
            sn.set(rc = {'figure.figsize':(10,10)})
            sn.heatmap(HLA_length_aatype_position_softmax_pd,
                       cmap = 'Blues',
                       cbar_kws={"shrink": 0.4},
                       square = True)
            if not length:
                length = '8-14mers'
            plt.title('{} | Peptide length:{} | Label:{} | Softmax Normalization'.format(hla, length, label))
            plt.show()
            
            sn.set(rc = {'figure.figsize':(10,10)})
            sn.heatmap(HLA_length_aatype_position_unsoftmax_pd,
                       cmap = 'Blues',
                       cbar_kws={"shrink": 0.4},
                       square = True)
            plt.title('{} | Peptide length:{} | Label:{} | Without Normalization'.format(hla, length, label))
            plt.show()

            
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd, max_list, new_pseudo
    
    else:
        HLA_length_aatype_position_pd = attn_HLA_length_aatype_position_pd_HLApseudo(HLA_length_aatype_position, 
                                                                           length, 
                                                                           hla_size,
                                                                           softmax,
                                                                           unsoftmax)
        HLA_length_aatype_position_pd = sort_aatype(HLA_length_aatype_position_pd)
        return HLA_length_aatype_position_pd, max_list, new_pseudo
    
    
def attn_HLA_length_aatype_position_pd_HLApseudo(HLA_length_aatype_position, df_size, softmax_dim = -1, length = 9, hla_size=34, n_heads = 3, softmax = True, unsoftmax = True):
        
    HLA_length_aatype_position_pd = np.zeros((20, hla_size))
    func = lambda x: round(x,4)
    
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in HLA_length_aatype_position.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            HLA_length_aatype_position_pd[aai, posi] = v
        aai += 1
    
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
    
    if softmax and not unsoftmax: 
        
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = softmax_dim)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, hla_size + 1))
        return HLA_length_aatype_position_softmax_pd
    
    elif unsoftmax and not softmax:
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, hla_size + 1))
        return HLA_length_aatype_position_unsoftmax_pd
    
    elif softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = softmax_dim)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, hla_size + 1))
        
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd/df_size,
                                                               index = aa_indexs, columns = range(1, hla_size + 1))
       
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd


    
def attn_HLA_length_aatype_position_num_HLAfull(data, attn_data, n_heads, hla, label = None, length = 9, show_num = False):
    aatype_position = dict()
    if label == None:
        if length == None:
            length_index = np.array(data[data.HLA == hla][data.label == label].index)
        else:
            length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)
    elif length == None:
            length_index = np.array(data[data.HLA == hla][data.label == label].index)
    else:
        length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)

        print("\n\nHLA and Label are filtered.")

    length_data_num = len(length_index)
    print("#Peptides:", length_data_num)
    print("HLA:", hla)
    for head in range(n_heads):
        cc = -1
        p = False
        for idx in length_index:
            p = False
            cc += 1
            temp_hla = data.iloc[idx].HLA_sequence
            temp_length_head_ = attn_data[idx][head]
            temp_length_head = temp_length_head_.sum(axis = 0)
            for i, aa in enumerate(temp_hla): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 
    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_hla = data.iloc[idx].HLA_sequence
            for i, aa in enumerate(temp_hla):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, length_data_num, aatype_position_num
    else:
        return aatype_position, length_data_num
    
    
def draw_hla_length_aatype_position_HLAfull(data, attn_data, n_heads, hla, pseudo, mhc_seq, label = None, length = 9, threshold = 0.95, top = True, show = True, softmax = True, unsoftmax = True, scale=False):
    
    func = lambda x: round(x,2)
    hla_size = len(mhc_seq) 
    HLA_length_aatype_position, length_data_num = attn_HLA_length_aatype_position_num_HLAfull(data, attn_data, n_heads, hla, label, length, show_num = False)
    
    if softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd_HLAfull(
                                                                                     HLA_length_aatype_position, 
                                                                                     length_data_num,
                                                                                     length,
                                                                                     hla_size,
                                                                                     softmax,
                                                                                     unsoftmax,
                                                                                     scale=False)

        HLA_length_aatype_position_softmax_pd = sort_aatype(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_unsoftmax_pd = sort_aatype(HLA_length_aatype_position_unsoftmax_pd)
        
        max_list = []
        hla_len = 0
        for (columnName, columnData) in HLA_length_aatype_position_unsoftmax_pd.iteritems():
            hla_len += 1
            value = np.max(columnData.values, axis=0)
            max_list.append(value)
            
        max_list = list(map(func, max_list))
        top_34_idx = np.argsort(max_list)[-34:]
        
        threshold_inner = round(np.max(np.array(max_list)) * threshold, 2)

        new_pseudo = []
        new_pseudo_pos = []
        new_pseudo_val = []
        
        pos = 0
        for (columnName, columnData) in HLA_length_aatype_position_unsoftmax_pd.iteritems():
            aa = list(blosum_dict.keys())[np.argmax(columnData.values, axis=0)][0]
            value = np.max(columnData.values, axis=0)
            if top:
                if pos in top_34_idx:
                    threshold_inner = value-1
                else:
                    threshold_inner = value+1
            aa = print_seq(aa, value, threshold_inner)
            if aa:
                new_pseudo.append(aa)
                new_pseudo_pos.append(pos)
                new_pseudo_val.append(value)
            pos += 1
        
        bin_pos = [7,9,24,45,59,62,63,66,67,69,75,69,70,73,74,77,80,81,84,95,97,99,114,116,118,143,150,152,156,158,159,163,167,171]
        bin__ = [0]*180
        for i in range(len(bin__)):
            if i in bin_pos:
                bin__[i-1] = 1
            else:
                bin__[i-1] = 0
        
        if show == 'bar':
            if top:
                threshold_inner = np.array(new_pseudo_val).min()
            print(color.END + "\nthreshold:", threshold_inner)
            
            sn.set(rc = {'figure.figsize':(30,10)})
            sn.set_style("white")
            ax = sn.barplot(x=np.arange(len(max_list)), y=max_list)
            #ax.bar_label(ax.containers[0])
            ax.set_xticklabels(mhc_seq)
            ax.axhline(threshold_inner)
            plt.legend(loc='upper right', labels=['MHC Pseudo-sequence'])
            idx = 0
            if hla_len > 34:
                for bar in ax.patches:
                    if bin__[idx] == 1:
                        bar.set_color('teal')    
                    else:
                        bar.set_color('lightblue')
                    idx += 1
                
            else:
                for bar in ax.patches:
                    if idx in new_pseudo_pos:
                        bar.set_color('teal')    
                    else:
                        bar.set_color('lightblue')
                    idx += 1
                  
                
            
            plt.show()
        if show == 'heatmap':
            sn.set(rc = {'figure.figsize':(15,15)})
            sn.heatmap(HLA_length_aatype_position_softmax_pd,
                       cmap = 'Blues',
                       cbar_kws={"shrink": 0.4},
                       square = True)
            plt.title('{} | Peptide length:{} | Label:{} | Softmax Normalization'.format(hla, length, label))
            plt.show()
            
            sn.set(rc = {'figure.figsize':(15,15)})
            sn.heatmap(HLA_length_aatype_position_unsoftmax_pd,
                       cmap = 'Blues',
                       cbar_kws={"shrink": 0.4},
                       square = True)
            plt.title('{} | Peptide length:{} | Label:{} | Without Normalization'.format(hla, length, label))
            plt.show()
            
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd, max_list, new_pseudo, top_34_idx
    
    else:
        HLA_length_aatype_position_pd = attn_HLA_length_aatype_position_pd_HLAfull(HLA_length_aatype_position, 
                                                                           length, 
                                                                           hla_size,
                                                                           softmax,
                                                                           unsoftmax)
        HLA_length_aatype_position_pd = sort_aatype(HLA_length_aatype_position_pd)
        return HLA_length_aatype_position_pd, max_list, new_pseudo, top_34_idx
    
    
def attn_HLA_length_aatype_position_pd_HLAfull(HLA_length_aatype_position, df_size, length = 9, hla_size=180, n_heads = 3, softmax = True, unsoftmax = True, scale=False):
        
    HLA_length_aatype_position_pd = np.zeros((20, hla_size))
    func = lambda x: round(x,4)
    
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in HLA_length_aatype_position.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            HLA_length_aatype_position_pd[aai, posi] = v
        aai += 1
    
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
    
    if softmax and not unsoftmax: 
        
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, hla_size + 1))
        return HLA_length_aatype_position_softmax_pd
    
    elif unsoftmax and not softmax:
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, hla_size + 1))
        return HLA_length_aatype_position_unsoftmax_pd
    
    elif softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, hla_size + 1))
        
        if scale:
            HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd/df_size,
                                                                   index = aa_indexs, columns = range(1, hla_size + 1))
        else:
            HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                                   index = aa_indexs, columns = range(1, hla_size + 1))            
       
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
    

def attn_HLA_length_aatype_position_num_HLA(data, attn_data, n_heads, hla = 'HLA-A*11:01', label = None, length = 9, show_num = False):
    aatype_position = dict()
    if label == None:
        length_index = np.array(data[data.length == length][data.HLA == hla].index)
    else:
        length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)

    length_data_num = len(length_index)
    print("\n\n#Peptides:", length_data_num)
    for head in range(n_heads):
        cc = -1
        p = False
        for idx in length_index:
            p = False
            cc += 1
            temp_hla = data.iloc[idx].HLA_sequence
            temp_length_head = deepcopy(nn.Softmax(dim=-1)(attn_data[idx][head])) 
            temp_length_head = nn.Softmax(dim=-1)(temp_length_head.sum(axis = 0))
            for i, aa in enumerate(temp_hla): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 
    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_hla = data.iloc[idx].HLA_sequence
            for i, aa in enumerate(temp_hla):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, aatype_position_num
    else:
        return aatype_position
    
def draw_hla_length_aatype_position_HLA(data, attn_data, n_heads, hla = 'HLA-B*27:05', label = None, length = 9, threshold = 0.95, show = True, softmax = True, unsoftmax = True):
    
    HLA_length_aatype_position = attn_HLA_length_aatype_position_num_HLA(data, attn_data, n_heads, hla, label, length, show_num = False)
    if softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd_HLA(
                                                                                     HLA_length_aatype_position, 
                                                                                     length, 
                                                                                     softmax,
                                                                                     unsoftmax)

        HLA_length_aatype_position_softmax_pd = sort_aatype(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_unsoftmax_pd = sort_aatype(HLA_length_aatype_position_unsoftmax_pd)
        
        max_list = []
        for (columnName, columnData) in HLA_length_aatype_position_softmax_pd.iteritems():
            aa = list(blosum_dict.keys())[np.argmax(columnData.values, axis=0)][0]
            value = np.max(columnData.values, axis=0)
            max_list.append(value)
            
        threshold_inner = round(np.max(np.array(max_list)) * threshold, 3)
        print("threshold:", threshold_inner)
        for (columnName, columnData) in HLA_length_aatype_position_softmax_pd.iteritems():
            aa = list(blosum_dict.keys())[np.argmax(columnData.values, axis=0)][0]
            value = np.max(columnData.values, axis=0)
            print_seq(aa, value, threshold_inner)
        
        if show:
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (25, 5))
            sn.heatmap(HLA_length_aatype_position_softmax_pd,
                        ax = axes[0], cmap = 'Blues', square = True)

            sn.heatmap(HLA_length_aatype_position_unsoftmax_pd,
                        ax = axes[1], cmap = 'Blues', square = True)

            axes[0].set_title('{} | Peptide length:{} | Label:{} | Softmax Normalization'.format(hla, length, label))
            axes[1].set_title('{} | Peptide length:{} | Label:{} | Without Normalization'.format(hla, length, label))
            plt.show()

        #print("HLA_length_aatype_position_softmax_pd", HLA_length_aatype_position_softmax_pd)
            
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
    
    else:
        HLA_length_aatype_position_pd = attn_HLA_length_aatype_position_pd_HLA(HLA_length_aatype_position, 
                                                                           length, 
                                                                           softmax,
                                                                           unsoftmax)
        HLA_length_aatype_position_pd = sort_aatype(HLA_length_aatype_position_pd)
        return HLA_length_aatype_position_pd
    
def attn_HLA_length_aatype_position_pd_HLA(HLA_length_aatype_position, length = 9, n_heads = 3, softmax = True, unsoftmax = True):
        
    HLA_length_aatype_position_pd = np.zeros((20, 34))
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in HLA_length_aatype_position.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            HLA_length_aatype_position_pd[aai, posi] = v
        aai += 1
    
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
    
    if softmax and not unsoftmax: 
        HLA_length_aatype_position_softmax_pd = deepcopy(nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd)))
        HLA_length_aatype_position_softmax_pd = np.array(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, 34 + 1))
        return HLA_length_aatype_position_softmax_pd
    
    elif unsoftmax and not softmax:
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, 34 + 1))
        return HLA_length_aatype_position_unsoftmax_pd
    
    elif softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd = deepcopy(nn.Softmax(dim = -1)(torch.Tensor(HLA_length_aatype_position_pd)))
        HLA_length_aatype_position_softmax_pd = np.array(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, 34 + 1))
        
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, 34 + 1))
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
    

    
def attn_HLA_length_aatype_position_num(data, attn_data, n_heads, hla, label = None, length = 9, show_num = False):

    aatype_position = dict()
    
    if corr:
        
        data['bin_'] = [round(x) for x in list(data['pred_'])]
        if hla == None: 
            if length == None:
                length_index = np.array(data[data.label == label][data['target_']==data['bin_']].index)
            else:
                length_index = np.array(data[data.label == label][data.length ==length][ data['target_']==data['bin_'] ].index)
                print(length_index)
       
    else:
        if hla == None: 
            if length == None:
                length_index = np.array(data[data.label == label].index)
            else:
                length_index = np.array(data[data.label == label][data.length == length].index)

        elif label == None:
            if length == None:
                length_index = np.array(data[data.HLA == hla][data.label == label].index)
            else:
                length_index = np.array(data[data.length == length][data.HLA == hla].index)
        elif length == None and hla == None:
                length_index = np.array(data[data.label == label].index)
        elif length == None and hla == None and label == None:
                length_index = np.array(data.index)
        elif length == None:
                length_index = np.array(data[data.HLA == hla][data.label == label].index)
        else:
            length_index = np.array(data[data.length == length][data.HLA == hla][data.label == label].index)
               
        print("\n\nHLA and Label are filtered.")

    length_data_num = len(length_index)
    print("#Peptides:", length_data_num)
    print("HLA:", hla)
    for head in range(n_heads):
        cc = -1
        p = False
        for idx in length_index:
            p = False
            cc += 1
            temp_peptide = data.iloc[idx].peptide
            length = len(temp_peptide)
            temp_length_head = attn_data[idx][head][:, :length]
            temp_length_head = nn.Softmax(dim=-1)(temp_length_head.sum(axis = 0))
            for i, aa in enumerate(temp_peptide): 
                aatype_position.setdefault(aa, {})
                aatype_position[aa].setdefault(i, 0)
                aatype_position[aa][i] += temp_length_head[i] 

    if show_num:
        aatype_position_num = dict()
        for idx in length_index:
            temp_peptide = data.iloc[idx].peptide
            for i, aa in enumerate(temp_peptide):
                aatype_position_num.setdefault(aa, {})
                aatype_position_num[aa].setdefault(i, 0)
                aatype_position_num[aa][i] += 1
             
        return aatype_position, length_data_num, aatype_position_num
    else:
        return aatype_position, length_data_num
    
    
def draw_hla_length_aatype_position(data, attn_data, hla, n_heads, label = None, length = 9, threshold = 0.95, top = True, show = True, softmax = True, unsoftmax = True):
    

    func = lambda x: round(x,2)
    HLA_length_aatype_position, length_data_num = attn_HLA_length_aatype_position_num(data, attn_data, n_heads, hla, label, length, show_num = False)
    
    if softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd = attn_HLA_length_aatype_position_pd(
                                                                                     HLA_length_aatype_position, 
                                                                                     length_data_num,
                                                                                     length,
                                                                                     softmax,
                                                                                     unsoftmax)

        HLA_length_aatype_position_softmax_pd = sort_aatype(HLA_length_aatype_position_softmax_pd)
        HLA_length_aatype_position_unsoftmax_pd = sort_aatype(HLA_length_aatype_position_unsoftmax_pd)
        
        max_list = []
        hla_len = 0
        for (columnName, columnData) in HLA_length_aatype_position_unsoftmax_pd.iteritems():
            hla_len += 1
            value = np.max(columnData.values, axis=0)
            max_list.append(value)
            
        max_list = list(map(func, max_list))
        
        threshold_inner = round(np.max(np.array(max_list)) * threshold, 2)
        print("Threshold:", threshold_inner)
        new_pseudo = []
        new_pseudo_pos = []
        new_pseudo_val = []
        
        pos = 0
        if length == None:
            length = 14
        
        for (columnName, columnData) in HLA_length_aatype_position_unsoftmax_pd.iteritems():
            aa = list(blosum_dict.keys())[np.argmax(columnData.values, axis=0)][0]
            value = np.max(columnData.values, axis=0)
            aa = print_seq(aa, value, threshold_inner)
            if aa:
                new_pseudo.append(aa)
                new_pseudo_pos.append(pos)
                new_pseudo_val.append(value)
            pos += 1
        
        if show:
            sn.set(rc = {'figure.figsize':(10,10)})
            sn.heatmap(HLA_length_aatype_position_softmax_pd,
                       cmap = sn.cubehelix_palette(as_cmap=True),
                       cbar_kws={"shrink": 0.4},
                       square = True)
            if not length:
                length = '8-14mers'
            plt.title('{} | Peptide length:{} | Label:{} | Softmax Normalization'.format(hla, length, label))
            plt.show()
            
            sn.set(rc = {'figure.figsize':(10,10)})
            sn.heatmap(HLA_length_aatype_position_unsoftmax_pd,
                       cmap = sn.cubehelix_palette(as_cmap=True),
                       cbar_kws={"shrink": 0.4},
                       square = True)
            plt.title('{} | Peptide length:{} | Label:{} | Without Normalization'.format(hla, length, label))
            plt.show()

            
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd, max_list, new_pseudo
    
    else:
        HLA_length_aatype_position_pd = attn_HLA_length_aatype_position_pd(HLA_length_aatype_position, 
                                                                           length, 
                                                                           softmax,
                                                                           unsoftmax)
        HLA_length_aatype_position_pd = sort_aatype(HLA_length_aatype_position_pd)
        return HLA_length_aatype_position_pd, max_list, new_pseudo
    
    
def attn_HLA_length_aatype_position_pd(HLA_length_aatype_position, df_size, length = 9, n_heads = 3, softmax = True, unsoftmax = True):
        
    if length == None:
        length = 14

    HLA_length_aatype_position_pd = np.zeros((20, length))
    func = lambda x: round(x,4)
    
    aai, aa_indexs = 0, []
    for aa, aa_posi in HLA_length_aatype_position.items():
        aa_indexs.append(aa)
        for posi, v in aa_posi.items():
            HLA_length_aatype_position_pd[aai, posi] = v
        aai += 1
    
    if len(aa_indexs) != 20: 
        aatype_sorts = list('ARNDCQEGHILKMFPSTWYV')
        abscent_aa = list(set(aatype_sorts).difference(set(aa_indexs)))
        aa_indexs += abscent_aa
    
    if softmax and not unsoftmax: 
        
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = 0)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, length + 1))
        return HLA_length_aatype_position_softmax_pd
    
    elif unsoftmax and not softmax:
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd,
                                                               index = aa_indexs, columns = range(1, length + 1))
        return HLA_length_aatype_position_unsoftmax_pd
    
    elif softmax and unsoftmax:
        HLA_length_aatype_position_softmax_pd = [ x.tolist() for x in nn.Softmax(dim = 0)(torch.Tensor(HLA_length_aatype_position_pd/df_size))]
        HLA_length_aatype_position_softmax_pd = np.array([list(map(func, i)) for i in HLA_length_aatype_position_softmax_pd])
        HLA_length_aatype_position_softmax_pd = pd.DataFrame(HLA_length_aatype_position_softmax_pd, 
                                                             index = aa_indexs, columns = range(1, length + 1))
        
        HLA_length_aatype_position_unsoftmax_pd = pd.DataFrame(HLA_length_aatype_position_pd/df_size,
                                                               index = aa_indexs, columns = range(1, length + 1))
       
        return HLA_length_aatype_position_softmax_pd, HLA_length_aatype_position_unsoftmax_pd
