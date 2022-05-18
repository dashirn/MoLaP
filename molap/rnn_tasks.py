#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:04:28 2021

@author: root
"""
import jax.numpy as np
# from jax.nn import normalize
import numpy as npo
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers
from jax.config import config
config.update("jax_enable_x64", True)
key = random.PRNGKey(0)
from jax import ops
import pickle
import matplotlib.pyplot as plt
plt.style.use('/Users/davidavinci/Documents/GitHub/prettyplots/prettyplots.mplstyle') # change the path as needed    


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from scipy.linalg import eigh

from sklearn.cross_decomposition import CCA

#Daniel Task dependencies & Parameters
# import sys
# sys.path.insert(0,'/Users/davidavinci/Documents/MountSinai/Code/PerturbativeRNN/DanielTask/forChristian')

# import taskMod312
# import dantools
# import time

# lrudKwargs = {
#     "hint_multiplier": 0,  # 0 hint multiplier = no hint
#     "val": 0.5,  # input pulse amplitude
#     "target_val": .5,  # target value amplitude
#     "delayPeriod": 0,  # time between evidence period and target
#     "cuePeriod": 80,#initially 2400  # time length (dt) of evidence period 
#     "duration": 5,#initially 100 # time length (dt) of input pulses
#     "answerTime": 20,#initially 500  # time length (dt) of target period
#     "numChoices": 2,  # number of output options
#     "output_mode": "plus_minus",  # plus_minus: target is positive or negative
#     # this only works with numChoices = 2
#     # multi_channel: positive targets but with multiple
#     # channels. this works for any numChoices
#     "goCue": True,  # Whether or not to use a go cue
#     "parameter_mode": "poisson_process",  # how to generate trials
#     "poisson_lambda": [
#         1 / 30,#75 #originally 1/300
#         1 / 80,#250 #originally 1/1000
#     ],  # each channel's poisson rate. shuffled before
#     # each task generation
# }

# # lrudKwargs = {
# #     "hint_multiplier": 0,  # 0 hint multiplier = no hint
# #     "val": 0.5,  # input pulse amplitude
# #     "target_val": .5,  # target value amplitude
# #     "delayPeriod": 0,  # time between evidence period and target
# #     "cuePeriod": 110,#initially 2400  # time length (dt) of evidence period 
# #     "duration": 10,#initially 100 # time length (dt) of input pulses
# #     "answerTime": 25,#initially 500  # time length (dt) of target period
# #     "numChoices": 2,  # number of output options
# #     "output_mode": "plus_minus",  # plus_minus: target is positive or negative
# #     # this only works with numChoices = 2
# #     # multi_channel: positive targets but with multiple
# #     # channels. this works for any numChoices
# #     "goCue": True,  # Whether or not to use a go cue
# #     "parameter_mode": "poisson_process",  # how to generate trials
# #     "poisson_lambda": [
# #         1 / 50,#originally 1/300
# #         1 / 80,#originally 1/1000
# #     ],  # each channel's poisson rate. shuffled before
# #     # each task generation
# # }

# lrud = taskMod312.LRUD_decision(**lrudKwargs)
# task = taskMod312.Task(lrud, [{}])
#########################################################

def save_figs_multiformats(save_path,save_name):
    for format_type in ['pdf','png','eps','svg']:
        plt.savefig(save_path + save_name+'.'+format_type, dpi=300, 
                    facecolor='w',edgecolor='w',orientation='portrait',
                    format=format_type)

def load_daniel_task_batch(batch_size,input_params,do_plot=False,do_save=False):
    key,bias_val, stddev_val, T, ntime, file_loc = input_params
    
    #Generate random int to load random batch
    nexamples = 1000
    curr_batch = random.randint(key,(1,),0,nexamples)
    
    file_loc = file_loc + '%d'%(curr_batch) +'.pickle'
    file_batch = open(file_loc,'rb')
    inputs = pickle.load(file_batch)
    targets = pickle.load(file_batch)
    file_batch.close()  
    
    return inputs,targets
    # return
    
def build_integrationtask(ntrials, input_params, do_plot=False,do_save=False):
    key,bias_val, stddev_val, T, ntime, save_name = input_params
    dt = T/float(ntime)
    nwninputs = 2
    biases_1xexw = np.expand_dims(bias_val * 2.0 * (npo.random.rand(ntrials,nwninputs) -0.5), axis=0)
    stddev = stddev_val / np.sqrt(dt)
    noise_txexw = stddev * npo.random.randn(ntime, ntrials, nwninputs)
    white_noise_txexw = biases_1xexw + noise_txexw
    # white_noise_txexw[0:5,:,:] =   0.0 # Allow context to establish before integration.
    white_noise_txexw = ops.index_update(white_noise_txexw,ops.index[0:5,
                                                                     np.shape(white_noise_txexw)[0],
                                                                     np.shape(white_noise_txexw)[1]],0)
    
    context_ex1 = np.expand_dims(npo.random.randint(0,2, ntrials), axis=1)
    context_ex2 = np.concatenate((context_ex1.astype(np.float64), 
                                  np.logical_not(context_ex1).astype(np.float64)), 
                                 axis=1)
    context_txex2 = np.repeat(np.expand_dims(context_ex2, axis=0), ntime, axis=0)

    inputs_txexu = np.concatenate((white_noise_txexw, context_txex2), axis=2)

    # Create the desired outputs
    # * dt, intentionally left off to get output scaling in O(1).
    int_wninputs_txexw = np.cumsum(white_noise_txexw, axis=0) 
    targets_txe = np.where(context_txex2[:,:,0], 
                           int_wninputs_txexw[:,:,0], 
                           int_wninputs_txexw[:,:,1])
    targets_txexm = np.expand_dims(targets_txe, axis=2) # single output, so m=1

    if do_plot:
        eidx = 0
        time = np.linspace(0, T, ntime)
        plt.figure(figsize=(13,6))
        plt.subplot(221)
        plt.plot(time, inputs_txexu[:,eidx,0:2]);
        plt.xlim([0,T])
        plt.ylabel('White-noise Inputs')
        plt.subplot(222)
        plt.plot(time, int_wninputs_txexw[:,eidx,:]);
        plt.xlim([0,T])
        plt.ylabel('Integrated Inputs')
        plt.subplot(223)
        plt.plot(time, inputs_txexu[:,eidx,2:])
        plt.xlim([0,T])
        plt.ylabel('Context')
        plt.subplot(224)
        plt.plot(time, targets_txexm[:,eidx,0], 'r')
        plt.title('Context {}'.format(context_txex2[0,eidx,:]))
        plt.xlim([0,T])
        plt.xlabel('Time')
        plt.ylabel('Output')

    #SAVE FIGS
    if do_save: save_figs_multiformats(save_name, 'Inputs&Targets_INTEGRATION_MANTE')      
        
    plt.show()
        
    return inputs_txexu, targets_txexm


def draw_noise_input(task_params,rkey):
    """

    Args:
        ntime (int tuple): parameter settings
        rkey (int): Jax specific key to draw noise

    Returns:
        noise_input_pluszero (TYPE): adds noise to inputs using rkey

    """
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    noise_input = stddev * random.normal(rkey,(ntime, batch_ntrials, num_inputs))
    noise_input_pluszero = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,\
                                                     num_inputs)),noise_input))
    return noise_input_pluszero

def create_angle_vecs(task_params,angles,rkey):
    """
    

    Args:
        params (tuple)
        angles (numpy array): array of angles at certaing spacing generated by 
        build_dunckernips_task_setup.
        rkey (int): Jax specific random key.

    Returns:
        cos_angles (numpy array): randomly drawn cos of angles
        sin_angles (numpy array): randomly drawn sin of angles.

    """
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles,predictive_interval, n_components_pca, do_plot, do_save = task_params
    angle_rand_idx = random.randint(rkey,(batch_ntrials,),0,np.size(angles))
    cos_angles = np.array([np.cos(np.radians(angles[i])) for i in angle_rand_idx])
    sin_angles = np.array([np.sin(np.radians(angles[i])) for i in angle_rand_idx])
    cos_angles = np.expand_dims(cos_angles,axis=0)
    cos_angles = np.expand_dims(cos_angles,axis=2)
    sin_angles = np.expand_dims(sin_angles,axis=0)
    sin_angles = np.expand_dims(sin_angles,axis=2)
    return cos_angles,sin_angles


def build_dunckernips_tasks_setup(task_setup_params):
    """
    Every task and every input in a particular task has independent
    random noise. => Every task uses different batch_subkey
    Currently uses the same rand key across all tasks to draw angles, 
    draw noise for rule/fixation inputs (subkey=0),
    and to shuffle trials, 

    Args:
        batch_ntrials (int): number trials in current batch.
        input_params (tuple): input parameters.
        rand_generator_idx (int): starting point for rand num generator.
        do_plot (bool, flag): Generate plots. Defaults to False.
        do_save (bool, flag): Save results. Defaults to False.

    Returns:
        task_params (tuple): task generating parameters.

    """
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                    dim_reduction_flag, do_plot, do_save = task_setup_params
    
    #Generate a particular new random key for a batch to get fresh batch
    batch_key = random.PRNGKey(rand_key_idx_task)
    batch_subkeys = []
    for k in range(30): 
        batch_key,subkey = random.split(batch_key)
        batch_subkeys.append(subkey)
    
    dt = T/float(ntime)
    stddev = sval / np.sqrt(dt)
    zeros_beginning = 10
    num_inputs = 1
    batch_ntrials = batch_size
    angles = np.arange(0,360,10)
    predictive_interval = 40 #max is 25 as stimulus appears at 25
    
    #Put all task relevant param settings into task_params tuple
    task_params = (batch_subkeys, ntime, zeros_beginning, num_inputs, 
                   batch_ntrials, dt, stddev, angles, predictive_interval,\
                       n_components_pca,do_plot, do_save) 
    
    ####Generate general inputs
    ### FIXATION INPUT
    inputs_f = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.ones((np.int_(ntime*3/4),batch_ntrials,1)),
                              np.zeros((np.int_(ntime*(1/4)),batch_ntrials,1))),axis=0)
    
    inputs_f_plusnoise = inputs_f + draw_noise_input(task_params, batch_subkeys[0])

    ## RULE INPUT
    input_r_zero = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((ntime,batch_ntrials,1))),axis=0)
    input_r_zero_plusnoise = input_r_zero + draw_noise_input(task_params,batch_subkeys[0])
    input_r_one = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.ones((ntime,batch_ntrials,1))),axis=0)
    input_r_one_plusnoise = input_r_one + draw_noise_input(task_params,batch_subkeys[0]) 
    
    ##Gather all inputs into tuple
    inputs_general = (inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise)    
    
    
    return task_params, inputs_general

    
def build_delaypro_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    
    ## Generate inputs
    input_s_cos = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(cos_angles,np.int_(ntime*3/4),axis=0),
                                          np.ones((np.int_(ntime*3/4),batch_ntrials,1)))),axis=0)
    input_s_cos_plusnoise = input_s_cos + draw_noise_input(task_params,batch_subkeys[1])#
    
    input_s_sin = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(sin_angles,np.int_(ntime*3/4),axis=0),
                                          np.ones((np.int_(ntime*3/4),batch_ntrials,1)))),axis=0)  
    input_s_sin_plusnoise = input_s_sin + draw_noise_input(task_params,batch_subkeys[2])#3    
    
    inputs_s = (input_s_cos_plusnoise,input_s_sin_plusnoise)
    # print(np.shape(input_s_cos_plusnoise))
    ## Generate targets
    target_cos = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(cos_angles,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)
    target_sin = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(sin_angles,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)    
    targets_s = (target_cos,target_sin)
    
    inputs = np.concatenate((inputs_f_plusnoise,
                                inputs_s[0],
                                inputs_s[1]),axis=2)
    # print(np.shape(inputs))
    targets = np.concatenate((inputs_f,
                                targets_s[0],
                                targets_s[1]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    # print(np.shape(inputs))
    # print(np.shape(targets))
    
    return inputs, targets

def build_delayanti_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles_opposite,sin_angles_opposite = create_angle_vecs(task_params,angles + 180,batch_subkeys[0]) 
    
    ## Generate inputs
    input_s_cos = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(cos_angles,np.int_(ntime*3/4),axis=0),
                                          np.ones((np.int_(ntime*3/4),batch_ntrials,1)))),axis=0)
    input_s_cos_plusnoise = input_s_cos + draw_noise_input(task_params,batch_subkeys[3])
    
    input_s_sin = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(sin_angles,np.int_(ntime*3/4),axis=0),
                                          np.ones((np.int_(ntime*3/4),batch_ntrials,1)))),axis=0)  
    input_s_sin_plusnoise = input_s_sin + draw_noise_input(task_params,batch_subkeys[4])  
    
    inputs_s = (input_s_cos_plusnoise,input_s_sin_plusnoise)
    
    ## Generate targets
    target_cos_opposite = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(cos_angles_opposite,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)
    target_sin_opposite = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(sin_angles_opposite,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)    
    targets_opposite = (target_cos_opposite,target_sin_opposite) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                                inputs_s[0],
                                inputs_s[1]),axis=2)
    targets = np.concatenate((inputs_f,
                                targets_opposite[0],
                                targets_opposite[1]),axis=2)  
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets    

def build_mempro_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles_opposite,sin_angles_opposite = create_angle_vecs(task_params,angles + 180,batch_subkeys[0]) 
    
    ## Generate inputs
    duration_val_stim = np.int_(ntime*2/4)
    duration_val_delay = np.int_(ntime*0)
    
    input_s_cos_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                          np.ones((duration_val_stim,batch_ntrials,1))),
                              np.zeros((duration_val_delay,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_plusnoise = input_s_cos_mem + draw_noise_input(task_params,batch_subkeys[5])
    
    input_s_sin_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                          np.ones((duration_val_stim,batch_ntrials,1))),
                              np.zeros((duration_val_delay,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)  
    input_s_sin_mem_plusnoise = input_s_sin_mem + draw_noise_input(task_params,batch_subkeys[6])  
    
    inputs_s_mem = (input_s_cos_mem_plusnoise,input_s_sin_mem_plusnoise)
    
    ## Generate targets
    target_cos = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(cos_angles,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)
    target_sin = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(sin_angles,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)    
    targets_s = (target_cos,target_sin)
    
    # inputs = np.concatenate((inputs_f_plusnoise,
    #                           inputs_s_mem[0],
    #                           inputs_s_mem[1],
    #                             input_r_zero_plusnoise,
    #                             input_r_zero_plusnoise,
    #                             input_r_one_plusnoise,
    #                             input_r_zero_plusnoise),axis=2)
    inputs = np.concatenate((inputs_f_plusnoise,
                              inputs_s_mem[0],
                              inputs_s_mem[1]),axis=2)
    targets = np.concatenate((inputs_f,
                              targets_s[0],
                              targets_s[1]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets 

def build_memanti_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles_opposite,sin_angles_opposite = create_angle_vecs(task_params,angles + 180,batch_subkeys[0]) 
    
    ## Generate inputs
    duration_val_stim = np.int_(ntime*2/4)
    duration_val_delay = np.int_(ntime*0)
    
    input_s_cos_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                          np.ones((duration_val_stim,batch_ntrials,1))),
                              np.zeros((duration_val_delay,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_plusnoise = input_s_cos_mem + draw_noise_input(task_params,batch_subkeys[5])
    
    input_s_sin_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                              np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                          np.ones((duration_val_stim,batch_ntrials,1))),
                              np.zeros((duration_val_delay,batch_ntrials,1)),
                              np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)  
    input_s_sin_mem_plusnoise = input_s_sin_mem + draw_noise_input(task_params,batch_subkeys[6])  
    
    inputs_s_mem = (input_s_cos_mem_plusnoise,input_s_sin_mem_plusnoise)
    
    ## Generate targets
    target_cos_opposite = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(cos_angles_opposite,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)
    target_sin_opposite = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                 np.zeros((np.int_(3/4*ntime),batch_ntrials,1)),
                                 np.multiply(np.repeat(sin_angles_opposite,np.int_(1/4*ntime),axis=0),
                                             np.ones((np.int_(1/4*ntime),batch_ntrials,1)))),axis=0)    
    targets_opposite = (target_cos_opposite,target_sin_opposite) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                                inputs_s_mem[0],
                                inputs_s_mem[1]),axis=2)
    targets = np.concatenate((inputs_f,
                                targets_opposite[0],
                                targets_opposite[1]),axis=2)  
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets

    
def build_memdm1_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
        
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles2,sin_angles2 = create_angle_vecs(task_params,angles,batch_subkeys[8]) 
    target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
    target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
    
    ## Generate inputs
    duration_val_stim = np.int_(10)#
    duration_val_interstim = 2*duration_val_stim
    duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
    input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(task_params,batch_subkeys[9])
    
    input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(task_params,batch_subkeys[10])
    
    inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                
      
    ###### GENERAL TASK TARGETS
    target_zeros = ntime - 2*np.int_(ntime*1/4)
    target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    
    target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                     cos_angles2+sin_angles2,cos_angles+sin_angles)
    targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0) 
    targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                                inputs_s_mem_dm[0]),axis=2)
    targets = np.concatenate((inputs_f,
                                targets_s_mem_dm[0]),axis=2) 
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets

def build_memdm2_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
        
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles2,sin_angles2 = create_angle_vecs(task_params,angles,batch_subkeys[11]) 
    target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
    target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
    
    ## Generate inputs
    duration_val_stim = np.int_(10)#
    duration_val_interstim = 2*duration_val_stim
    duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
    input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(task_params,batch_subkeys[12])
    
    input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(task_params,batch_subkeys[13])
    
    inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                
      
    ###### GENERAL TASK TARGETS
    target_zeros = ntime - 2*np.int_(ntime*1/4)
    target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    
    target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                     cos_angles2+sin_angles2,cos_angles+sin_angles)
    targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0) 
    targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                                inputs_s_mem_dm[1]),axis=2)
    targets = np.concatenate((inputs_f,
                                targets_s_mem_dm[1]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets

def build_contextmemdm1_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
        
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles2,sin_angles2 = create_angle_vecs(task_params,angles,batch_subkeys[14]) 
    target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
    target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
    
    ## Generate inputs
    duration_val_stim = np.int_(10)#
    duration_val_interstim = 2*duration_val_stim
    duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
    input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(task_params,batch_subkeys[15])
    
    input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(task_params,batch_subkeys[16])
    
    inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                
      
    ###### GENERAL TASK TARGETS
    target_zeros = ntime - 2*np.int_(ntime*1/4)
    target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    
    target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                     cos_angles2+sin_angles2,cos_angles+sin_angles)
    targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0) 
    targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                               inputs_s_mem_dm[0],
                                inputs_s_mem_dm[1]),axis=2)
    targets= np.concatenate((inputs_f,
                             targets_s_mem_dm[0]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets
    
def build_contextmemdm2_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
        
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles2,sin_angles2 = create_angle_vecs(task_params,angles,batch_subkeys[17]) 
    target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
    target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
    
    ## Generate inputs
    duration_val_stim = np.int_(10)#
    duration_val_interstim = 2*duration_val_stim
    duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
    input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(task_params,batch_subkeys[18])
    
    input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(task_params,batch_subkeys[19])
    
    inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                
      
    ###### GENERAL TASK TARGETS
    target_zeros = ntime - 2*np.int_(ntime*1/4)
    target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    
    target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                     cos_angles2+sin_angles2,cos_angles+sin_angles)
    targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0) 
    targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                               inputs_s_mem_dm[0],
                                inputs_s_mem_dm[1]),axis=2)
    targets = np.concatenate((inputs_f,
                              targets_s_mem_dm[1]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets

def build_multimem_task (task_params):
    
    #Build task params
    task_params, inputs_general = build_dunckernips_tasks_setup(task_params)
        
    batch_subkeys, ntime, zeros_beginning, num_inputs, batch_ntrials, dt, stddev,\
        angles, predictive_interval, n_components_pca, do_plot, do_save = task_params
    inputs_f,inputs_f_plusnoise,input_r_zero_plusnoise,input_r_one_plusnoise = inputs_general
    
    cos_angles,sin_angles = create_angle_vecs(task_params,angles,batch_subkeys[0]) 
    cos_angles2,sin_angles2 = create_angle_vecs(task_params,angles,batch_subkeys[20]) 
    target_vals_cos = np.where(cos_angles2>=cos_angles,cos_angles2,cos_angles)
    target_vals_sin = np.where(sin_angles2>=sin_angles,sin_angles2,sin_angles)
    
    ## Generate inputs
    duration_val_stim = np.int_(10)#
    duration_val_interstim = 2*duration_val_stim
    duration_val_delay = ntime - 2*duration_val_stim - duration_val_interstim - 2*np.int_(ntime*1/4)
    
    input_s_cos_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(cos_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_cos_mem_short_plusnoise = input_s_cos_mem_short + draw_noise_input(task_params,batch_subkeys[21])
    
    input_s_sin_mem_short = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles,duration_val_stim,axis=0),
                                      np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_interstim,batch_ntrials,1)),
                          np.multiply(np.repeat(sin_angles2,duration_val_stim,axis=0),
                          np.ones((duration_val_stim,batch_ntrials,1))),
                          np.zeros((duration_val_delay,batch_ntrials,1)),
                          np.zeros((np.int_(ntime*1/4),batch_ntrials,1))),axis=0)
    input_s_sin_mem_short_plusnoise = input_s_sin_mem_short + draw_noise_input(task_params,batch_subkeys[22])
    
    inputs_s_mem_dm = (input_s_cos_mem_short_plusnoise,input_s_sin_mem_short_plusnoise)
                
      
    ###### GENERAL TASK TARGETS
    target_zeros = ntime - 2*np.int_(ntime*1/4)
    target_s_cos_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_cos,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    target_s_sin_mem_long = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_sin,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0)
    
    target_vals_multi_mem = np.where(cos_angles2+sin_angles2>=cos_angles+sin_angles,
                                     cos_angles2+sin_angles2,cos_angles+sin_angles)
    targets_s_multi_mem = np.concatenate((np.zeros((zeros_beginning,batch_ntrials,1)),
                                            np.zeros((np.int_(ntime*1/4),batch_ntrials,1)),
                                            np.zeros((target_zeros,batch_ntrials,1)),
                                            np.multiply(np.repeat(target_vals_multi_mem,np.int_(ntime*1/4),axis=0),
                                                        np.ones((np.int_(ntime*1/4),batch_ntrials,1)))),axis=0) 
    targets_s_mem_dm = (target_s_cos_mem_long,target_s_sin_mem_long,targets_s_multi_mem) 
    
    inputs = np.concatenate((inputs_f_plusnoise,
                               inputs_s_mem_dm[0],
                                inputs_s_mem_dm[1]),axis=2)
    targets = np.concatenate((inputs_f,
                              targets_s_mem_dm[2]),axis=2)
    
    inputs = np.array(inputs)
    targets = np.array(targets)

    #Shuffle elements
    shuffle_idx = random.randint(batch_subkeys[0],(batch_ntrials,),0,batch_ntrials)
    inputs = inputs[:,shuffle_idx,:]
    targets = targets[:,shuffle_idx,:]
    
    return inputs, targets

def aggregate_tasks(input_params,leave_out=[]):
    
    inputs_stacked = []
    targets_stacked = []
    
    if 'delay_pro' not in leave_out:
        inputs, targets = build_delaypro_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"DelayPro Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'delay_anti' not in leave_out:
        inputs, targets = build_delayanti_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"DelayAnti Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'mem_pro' not in leave_out:
        inputs, targets = build_mempro_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"MemPro Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'mem_anti'  not in leave_out:
        inputs, targets = build_memanti_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"MemAnti Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'mem_dm1'  not in leave_out:
        inputs, targets = build_memdm1_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"MemDm1 Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'mem_dm2'  not in leave_out:
        inputs, targets = build_memdm2_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"MemDm2 Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'context_mem_dm1' not in leave_out:
        inputs, targets = build_contextmemdm1_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"ContextMemDm1 Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'context_mem_dm2'  not in leave_out:
        inputs, targets = build_contextmemdm2_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"ContextMemDm2 Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    if 'multi_mem'  not in leave_out:
        inputs, targets = build_multimem_task(input_params)
        inputs_stacked.append(np.array(inputs))
        targets_stacked.append(np.array(targets))
        # print(f"MultiMem Task. Input shape {np.shape(inputs)}. Target shape {np.shape(targets)}")
    
    #Get arrays into proper shape for PCA
    inputs_stacked = np.concatenate((inputs_stacked),axis=2)
    targets_stacked = np.concatenate((targets_stacked),axis=2)
    
    
    return inputs_stacked, targets_stacked


def transform_data_dimreduce(inputs,targets,direction_indicator,shape_info=None):
    
    if shape_info is None:
        inputs_shape = np.shape(inputs)
        targets_shape = np.shape(targets)
    else:
        inputs_shape,targets_shape = shape_info
    
    if direction_indicator == 1: #Transform for projection into PC space
        #Collapse batch dim
        inputs_transformed = np.reshape(inputs,(inputs_shape[0]*inputs_shape[1],inputs_shape[2]))
        targets_transformed = np.reshape(targets,(targets_shape[0]*targets_shape[1],targets_shape[2]))
        
        #Standardize
        inputs_transformed = StandardScaler().fit_transform(inputs_transformed)
        targets_transformed = StandardScaler().fit_transform(targets_transformed)
    elif direction_indicator == -1: #Transform post dim reduction for training

        #Rescale into 0-1 range
        min_val_in = np.min(inputs.reshape(-1,)); 
        max_val_in = np.max(inputs.reshape(-1,));
        inputs_transformed = np.array([(x-min_val_in) / (max_val_in - min_val_in) for x in inputs])
        
        min_val_out = np.min(targets.reshape(-1,)); 
        max_val_out = np.max(targets.reshape(-1,));
        targets_transformed = np.array([(x-min_val_out) / (max_val_out - min_val_out) for x in targets])
        
        #Reshape inputs,targets to recover batch dimension
        inputs_transformed = np.reshape(inputs_transformed,(inputs_shape[0],inputs_shape[1],-1))
        targets_transformed = np.reshape(targets_transformed,(targets_shape[0],targets_shape[1],-1))
            
    return inputs_transformed, targets_transformed
        

def get_cancor_components(inputs,targets,task_setup_params):
    """
    Transforms task inputs & targets into CCA space.

    Args:
        inputs (TYPE): DESCRIPTION.
        targets (TYPE): DESCRIPTION.
        task_setup_params (TYPE): DESCRIPTION.

    Returns:
        cca_results (dict): Transformed CCA inputs/targets and fitted CCA model.

    """
    print('Run CANCOR transform')
    
    #Get parameters
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                    dim_reduction_flag, do_plot, do_save = task_setup_params
                    
    original_inputs_shape = np.shape(inputs)
    original_targets_shape = np.shape(targets)
    original_shape_info = original_inputs_shape, original_targets_shape
    
    #Prepare data for dim reduction
    inputs_transformed, targets_transformed = transform_data_dimreduce(inputs,targets,1,original_shape_info)
    
    #Fit CCA
    ca = CCA(n_components=n_components_dimreduction)
    ca.fit(inputs_transformed,targets_transformed)
    
    inputs_cca,targets_cca = ca.transform(inputs_transformed,targets_transformed)
    
    
    # #Get R2 scores for all trials
    # r2_score = ca.score(inputs_transformed,targets_transformed)
    
    #Transform CCA-ed data back into original shape for training
    inputs_cca, targets_cca = transform_data_dimreduce(inputs_cca,targets_cca,-1,original_shape_info)
    
    # #Plot
    # plt.figure(figsize=(15,12))
    # plt.subplot(2,1,1)
    # plt.plot(inputs_cca)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Pre-training INPUTS')
    
    # plt.subplot(2,1,2)
    # plt.plot(targets_cca)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Pre-training TARGETS')
    
    # plt.tight_layout()
    # plt.show()
    
    cca_results = {}
    cca_results['inputs_cca'] = inputs_cca
    cca_results['targets_cca'] = targets_cca
    cca_results['cca_fit_model'] = ca
    # cca_results['r2'] = r2_score
    cca_results['original_inputs_shape'] = original_inputs_shape
    cca_results['original_targets_shape'] = original_targets_shape
    
    return cca_results

def get_pca_components(inputs,targets,task_setup_params):
    """
    

    Args:
        inputs (TYPE): task inputs (np array) .
        targets (TYPE): task targets (np array).
        task_setup_params (TYPE): parameters (tuple).

    Returns:
        Inputs & Targets PC transformed. As well as 
        eigenvals & eigenvacs for the space

    """
    print('PCA transform')
    
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                    dim_reduction_flag, do_plot, do_save = task_setup_params
                    
    original_inputs_shape = np.shape(inputs)
    original_targets_shape = np.shape(targets)
    original_shape_info = original_inputs_shape, original_targets_shape
    
    #Prepare data for dim reduction
    inputs_transformed, targets_transformed = transform_data_dimreduce(inputs,targets,1,original_shape_info)
    
    #Fit PCA to inputs & the use on targets    
    pca_results = {}
    pca_model = PCA(n_components_dimreduction)
    concatenated_data = np.concatenate((inputs_transformed,targets_transformed),axis=1)
    pca_model.fit(concatenated_data)
    pc_data = pca_model.transform(concatenated_data)
    # pca_out = PCA(n_components_dimreduction)
    # pca_out.fit(targets_transformed)
    
    #Transform CCA-ed data back into original shape for training
    inputs_pca, targets_pca = transform_data_dimreduce(pc_data[:,:original_inputs_shape[1]],
                                                       pc_data[:,-original_targets_shape[1]:],-1,original_shape_info)
    
    pca_results['inputs_pca'] = inputs_pca
    pca_results['original_inputs_shape'] = original_inputs_shape
    pca_results['targets_pca'] = targets_pca
    pca_results['original_targets_shape'] = original_targets_shape
    pca_results['pca_fit_model'] = pca_model


    return pca_results
