import os
import numpy as np
import random

# Function to EEG data, provided by EE C247 class
def load_data(dir_path): 
    X_test = np.load(os.path.join(dir_path,"X_test.npy"))
    X_test = np.expand_dims(X_test,axis=-1)

    y_test = np.load(os.path.join(dir_path,"y_test.npy"))
    y_test -= np.amin(y_test)

    person_train_valid = np.load(os.path.join(dir_path,"person_train_valid.npy"))

    X_train_valid = np.load(os.path.join(dir_path,"X_train_valid.npy"))
    X_train_valid = np.expand_dims(X_train_valid,axis=-1)

    y_train_valid = np.load(os.path.join(dir_path,"y_train_valid.npy"))
    y_train_valid -= np.amin(y_train_valid)

    person_test = np.load(os.path.join(dir_path,"person_test.npy"))

    return X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test

def split_data_by_subject(X,y,person): 
    '''
    Function to split data by subject
    INPUT: 
    X - EEG data (num_trials, num_eeg_electrodes, time_bins) 
    y - target data corresponding to correct motor imagery task (num_trials, )
    person - subject performing task, ranging from 0-8 (num_trials, )
    OUTPUT:
    X_subject - EEG data by subject (num_subject, num_trials, num_eeg_electrodes, time_bins)
    y_subject - target data by subject (num_subject, num_trials)
    person_separated - subject performing task separated for utility 
    '''
    
    # ensure y and person dims are correct
    if y.ndim == 2: y = np.squeeze(y)
    if person.ndim == 2: person = np.squeeze(person) 
        
    
    #split data in to subjects
    X_subject = [X[person==i] for i in range(9)]
    y_subject = [y[person==i] for i in range(9)]
    person_separated = [person[person==i] for i in range(9)] 
    
    return X_subject, y_subject, person_separated
    
def split_data_by_task(X,y,person): 
    '''
    Function that splits data by target task
    INPUT: 
    X - EEG data (num_trials, num_eeg_electrodes, time_bins) 
    y - target data corresponding to correct motor imagery task (num_trials, )
    person - subject performing task, ranging from 0-8 (num_trials, )
    OUTPUT:
    X_task - EEG data by task (num_task, num_trials, num_eeg_electrodes, time_bins)
    y_subject - target data by subject (num_task, num_trials)
    person_separated - subject performing task (num_task,num_trials) 
    '''
    
    # ensure y and person dims are correct
    if y.ndim == 2: y = np.squeeze(y)
    if person.ndim == 2: person = np.squeeze(person) 
        
    #split data in to tasks
    X_task = [X[y==i] for i in range(4)]
    y_task = [y[y==i] for i in range(4)]
    person_task = [person[y==i] for i in range(4)] 
    
    return X_task, y_task, person_task
    