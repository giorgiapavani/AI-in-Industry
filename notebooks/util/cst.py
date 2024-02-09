#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import datasets
import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras import backend as k
#import tensorflow_lattice as tfl

figsize=(9,3)


# ==============================================================================
# Loading data
# ==============================================================================


def load_cmapss_data(data_folder):
    # Read the CSV files
    fnames = ['train_FD001', 'train_FD002', 'train_FD003', 'train_FD004']
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    datalist = []
    nmcn = 0
    for fstem in fnames:
        # Read data
        data = pd.read_csv(f'{data_folder}/{fstem}.txt', sep=' ', header=None)
        # Drop the last two columns (parsing errors)
        data.drop(columns=[26, 27], inplace=True)
        # Replace column names
        data.columns = cols
        # Add the data source
        data['src'] = fstem
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique())
        # Generate RUL data
        cnts = data.groupby('machine')[['cycle']].count()
        cnts.columns = ['ftime']
        data = data.join(cnts, on='machine')
        data['rul'] = data['ftime'] - data['cycle']
        data.drop(columns=['ftime'], inplace=True)
        # Store in the list
        datalist.append(data)
    # Concatenate
    data = pd.concat(datalist)
    # Put the 'src' field at the beginning and keep 'rul' at the end
    data = data[['src'] + cols + ['rul']]
    # data.columns = cols
    return data

def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()
    
def plot_dataframe(data, labels=None, vmin=-1.96, vmax=1.96,
        figsize=None, s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect='auto',
            cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = - 0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                s=s,
                color=plt.get_cmap('tab10')(np.mod(labels, 10)))
    plt.tight_layout()
    
    
def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=None,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None):
    # Open a new figure
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=5);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=5)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2, s=5)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=5)
    plt.grid(linestyle=':')
    plt.tight_layout()
    
    
# ==============================================================================
# Data inspection and manipulation
# ==============================================================================

def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res


def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    if len(ts_list) > 0:
        ts_data = pd.concat(ts_list)
    else:
        ts_data = pd.DataFrame(columns=tr_data.columns)
    return tr_data, ts_data

def split_machines(data, supervised, unsupervised):
    np.random.seed(42)
    machines = data.machine.unique()
    np.random.shuffle(machines)

    sep_trs = int(supervised * len(machines))
    sep_tru = int(unsupervised * len(machines))
    
    
    if unsupervised == 0:
        trs_mcn = list(machines[:sep_trs]) #prende machines con dati supervisionati
        ts_mcn = list(machines[sep_trs:]) #restanti per test (25% del totale)
        print(f'Num. machines: {len(trs_mcn)} (supervised), {len(ts_mcn)} (test)')
        return partition_by_machine(data, trs_mcn)
    
    elif supervised == 0:
        tru_mcn = list(machines[:sep_tru]) #prende machines con dati unsupervisised
        ts_mcn = list(machines[sep_tru:]) #restanti per test (25% del totale)
        print(f'Num. machines: {len(tru_mcn)} (unsupervised), {len(ts_mcn)} (test)')
        return partition_by_machine(data, tru_mcn)
    
    else:
        trs_mcn = list(machines[:sep_trs]) #prende machines con dati supervisionati
        tru_mcn = list(machines[sep_trs:sep_tru]) #prende machines con dati non supervisionati
        ts_mcn = list(machines[sep_tru:]) #restanti per test (25% del totale)
        print(f'Num. machines: {len(trs_mcn)} (supervised), {len(tru_mcn)} (unsupervised), {len(ts_mcn)} (test)')
        tr, ts = partition_by_machine(data, trs_mcn + tru_mcn)
        trs, tru = partition_by_machine(tr, trs_mcn)
        return tr, ts, trs, tru
    
    
#function to standardize data and normalize rul values
def standardize(tr, ts, dt_in):
    trmean = tr[dt_in].mean() #NB dt_in selects columns with sensors data
    trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields

    tr_s = tr.copy()
    tr_s[dt_in] = (tr_s[dt_in] - trmean) / trstd #standardize train set 
    trmaxrul = tr_s['rul'].max()
    tr_s['rul'] = tr_s['rul'] / trmaxrul
    
    ts_s = ts.copy()
    ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd #standardize test set 
    ts_s['rul'] = ts_s['rul'] / trmaxrul
    
    return tr_s, ts_s, trmaxrul

def standardize_mixed(tr, trs, tru, ts, dt_in):
    trmean = tr[dt_in].mean() #NB dt_in selects columns with sensors data
    trstd = tr[dt_in].std().replace(to_replace=0, value=1) # handle static fields
    trmaxrul = tr['rul'].max()

    ts_s = ts.copy()
    ts_s[dt_in] = (ts_s[dt_in] - trmean) / trstd
    #tr[dt_in] = (tr[dt_in] - trmean) / trstd
    trs_s = trs.copy()
    trs_s[dt_in] = (trs_s[dt_in] - trmean) / trstd
    tru_s = tru.copy()
    tru_s[dt_in] = (tru_s[dt_in] - trmean) / trstd
    
    ts_s['rul'] = ts_s['rul'] / trmaxrul 
    #tr['rul'] = tr['rul'] / trmaxrul 
    trs_s['rul'] = trs_s['rul'] / trmaxrul
    tru_s['rul'] = tru_s['rul'] / trmaxrul
        
    return trs_s, tru_s, ts_s, trmaxrul
    

def evaluation(pred, ts_s):
    mse_seeds = []
    for i in pred:
        mse_seeds.append(mean_squared_error(ts_s["rul"], i)) # Calcola la media e la deviazione standard delle MSE
    mse_mean = np.mean(mse_seeds) 
    mse_std = np.std(mse_seeds) 
    print(f'Media della MSE: {mse_mean:.4f}\nDeviazione standard della MSE: {mse_std:.4f}')
    return mse_seeds, mse_mean, mse_std

def plot_results(mse_seeds, mse_mean, mse_std):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mse_seeds)), mse_seeds)
    plt.xticks(range(0,len(mse_seeds)))
    plt.axhline(y = mse_mean, color = 'r', linestyle = '--')
    plt.axhline(y = mse_mean + mse_std, color = 'y', linestyle = ':')
    plt.axhline(y = mse_mean - mse_std, color = 'y', linestyle = ':')
    plt.ylabel('MSE')
    plt.xlabel('Seed')
    plt.title('MSE distribution')
    plt.show()
    
    
def split_data(ts, trs=0, tru=0, trs_ratio=0, tru_ratio=0):
    if type(tru) == int: #only supervised data
        sep_trs = int(trs_ratio * len(trs))
        trs = trs[:sep_trs]
        print(f'Num. samples: {len(trs)} (supervised), {len(ts)} (test)')
        return trs
    
    elif type(trs) == int: #only unsupervised data
        sep_tru = int(tru_ratio * len(tru))
        tru = tru[:sep_tru]
        print(f'Num. samples: {len(tru)} (unsupervised), {len(ts)} (test)')
        return tru
    
    else: #mixed data
        sep_trs = int(trs_ratio * len(trs))
        sep_tru = int(tru_ratio * len(tru))
        trs = trs[:sep_trs]
        tru = tru[:sep_tru]
        print(f'Num. samples: {len(trs)} (supervised), {len(tru)} (unsupervised), {len(ts)} (test)')
        return trs, tru
    
    
def remove_rul(tru_s):
    tru_s_by_m = split_by_field(tru_s, 'machine')
    np.random.seed(42)
    for mcn, tmp in tru_s_by_m.items():  # mcn: macchina, tmp: dati di quella macchina
        cutoff = int(np.random.randint(10, 50, 1))    
        tru_s_by_m[mcn] = tmp.iloc[:-cutoff]
    tru_st = pd.concat(tru_s_by_m.values())
    tru_st['rul'] = -1.0 #we assign invalid rul values to unsupervised data
    return tru_st

# ==============================================================================
# Models and optimization
# ==============================================================================

class MLPRegressor(keras.Model):
    def __init__(self, input_shape, hidden=[]):
        super(MLPRegressor, self).__init__()
        # Build the model
        self.lrs = [layers.Dense(h, activation='relu') for h in hidden]
        self.lrs.append(layers.Dense(1, activation='linear'))

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x

#this function generates a list of batches for the training set, where each batch is a sequence of sorted samples from the same machine 
class CstBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, in_cols, batch_size, seed=42, validation_split=0.2):
        super(CstBatchGenerator).__init__()
        self.data = data
        self.in_cols = in_cols #dt_in (p, s and rul)
        self.dpm = split_by_field(data, 'machine')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.val_size = validation_split
        # Build the first sequence of batches
        self.__build_batches()

    def __len__(self):
        return len(self.batches)

    # def __getitem__(self, index):
    #     idx = self.batches[index]
    #     mcn = self.machines[index]
    #     x = self.data[self.in_cols].loc[idx].values
    #     y = self.data['rul'].loc[idx].values
    #     return x, y


    def __getitem__(self, index):
        idx = self.batches[index]
        # mcn = self.machines[index]
        x = self.data[self.in_cols].loc[idx].values
        y = self.data['rul'].loc[idx].values
        flags = (y != -1) #check if we have unsupervised data
        info = np.vstack((y, flags, idx)).T
        return x, info

    def on_epoch_end(self):
        self.__build_batches()

    def __build_batches(self):
        self.batches = []
        self.machines = []
        # Randomly sort the machines
        # self.rng.shuffle(mcns)
        # Loop over all machines
        mcns = list(self.dpm.keys())
        for mcn in mcns:
            # Obtain the list of indices
            index = self.dpm[mcn].index
            # Padding
            padsize = self.batch_size - (len(index) % self.batch_size)
            padding = self.rng.choice(index, padsize)
            idx = np.hstack((index, padding))
            # Shuffle
            self.rng.shuffle(idx)
            # Split into batches
            bt = idx.reshape(-1, self.batch_size)
            # Sort each batch individually
            bt = np.sort(bt, axis=1)
            # Store
            self.batches.append(bt)
            self.machines.append(np.repeat([mcn], len(bt)))
        # Concatenate all batches
        self.batches = np.vstack(self.batches)
        self.machines = np.hstack(self.machines)
        # Shuffle the batches
        bidx = np.arange(len(self.batches))
        self.rng.shuffle(bidx)
        self.batches = self.batches[bidx, :]
        self.machines = self.machines[bidx]
    

#this model extends an MLP regressor with a custom training step that includes a constraint regularization term in the loss function. 
# It is designed to handle data with unsupervised portions (indicated by flags) and aims to enforce smooth predictions over consecutive indices. 
# The alpha and beta parameters control the balance between the main loss and the constraint term during training.    
class CstRULRegressor(MLPRegressor):
    def __init__(self, input_shape, alpha, beta, maxrul, hidden=[]):
        super(CstRULRegressor, self).__init__(input_shape, hidden)
        # Weights - alpha and beta are coefficients controlling the balance between the main loss and the constraint regularization term
        self.alpha = alpha
        self.beta = beta
        self.maxrul = maxrul # is a normalization factor used in the constraint term
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst') #constraint term

    def train_step(self, data):
        x, info = data #get_item from batch generator
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self(x, training=True)
            # Compute the main loss (0 for unsupervised data)
            mse = k.mean(flags * k.square(y_pred-y_true))
            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = k.mean(k.square(deltadiff))
            loss = self.alpha * mse + self.beta * cst

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]
        

class CstPosRULRegressor(MLPRegressor):
    def __init__(self, input_shape, alpha, beta, gamma, maxrul, hidden=[]):
        super(CstPosRULRegressor, self).__init__(input_shape, hidden)
        # Weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # New hyperparameter for the positivity regularizer
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    # ... (rest of the class definition remains the same)

    def train_step(self, data):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        with tf.GradientTape() as tape:
            # Obtain the predictions
            y_pred = self(x, training=True)

            # Compute the main loss
            mse = k.mean(flags * k.square(y_pred - y_true))

            # Compute the constraint regularization term
            delta_pred = y_pred[1:] - y_pred[:-1]
            delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
            deltadiff = delta_pred - delta_rul
            cst = k.mean(k.square(deltadiff))

            # Additional regularization term for positivity
            positivity_regularizer = k.mean(self.gamma * k.square(k.maximum(0.0, -y_pred)))

            # Combine all regularization terms
            loss = self.alpha * mse + self.beta * cst + positivity_regularizer

        # Compute gradients
        tr_vars = self.trainable_variables
        grads = tape.gradient(loss, tr_vars)

        # Update the network weights
        self.optimizer.apply_gradients(zip(grads, tr_vars))

        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
    
        # Return additional regularization term for monitoring if needed
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result(),
                'positivity_regularizer': positivity_regularizer}
