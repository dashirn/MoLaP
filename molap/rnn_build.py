#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:25:59 2020

@author: cdmdc

Training context-switch task incrementally, with option to freeze weights 
in between

New: build_task_modular gives modular inputs/targets & uses jax random
number generator

Context switches between M1 (Up) with context = 1 and
M2 (Down) with context = -1
"""
#Imports
from __future__ import absolute_import
from __future__ import print_function
import jax.numpy as jnp
# from jax.nn import normalize
import numpy as np
from jax import jit,grad
# import numpy.random as npr
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental.optimizers import adam,make_schedule,optimizer
import pickle
# from scipy import signal
# import matplotlib.pyplot as plt

# Generate random key
from jax import random #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers
key = random.PRNGKey(0)
# from sklearn.preprocessing import StandardScaler

from rnn_tasks import aggregate_tasks,\
    build_delaypro_task,build_delayanti_task,build_mempro_task,build_memanti_task,\
            build_memdm1_task,build_memdm2_task,build_contextmemdm1_task,\
                build_contextmemdm2_task,build_multimem_task

from rnn_tasks import get_cancor_components, get_pca_components



################ RNN FUNCTIONS #############################################
    
def concat_multiply(weights, *args):
    cat_state = jnp.hstack(args + (jnp.ones((args[0].shape[0], 1)),))
    return jnp.dot(cat_state, weights)

def concat_args(*args):
    return jnp.hstack(args + (jnp.ones((args[0].shape[0], 1)),))

def rectilinear(x):
	return jnp.maximum(0.0, x)

def rnn_predict(params, inputs, return_hiddens=False):
    def update_rnn(input_item, hidden_units):
        return jnp.tanh(concat_multiply(params['change'], input_item, hidden_units))

    def hidden_to_output_probs(hidden_units):
        return concat_multiply(params['predict'], hidden_units)

    batch_size = inputs.shape[1]
    hidden_units = jnp.repeat(params['hidden unit'], batch_size, axis=0)
    outputs_time = []#outputs
    hiddens_time = []#hidden activations over time
    for input_item in inputs:  # Iterate over time steps.
        hidden_units = update_rnn(input_item, hidden_units)
        outputs_time.append(hidden_to_output_probs(hidden_units))
        if return_hiddens: hiddens_time.append(hidden_units)
    return jnp.array(outputs_time), jnp.array(hiddens_time)

def rnn_predict_rectilinear(params, inputs, return_hiddens=False):
    def update_rnn(input_item, hidden_units):
        return rectilinear(concat_multiply(params['change'], input_item, hidden_units))

    def hidden_to_output_probs(hidden_units):
        return concat_multiply(params['predict'], hidden_units)

    batch_size = inputs.shape[1]
    hidden_units = jnp.repeat(params['hidden unit'], batch_size, axis=0)
    outputs_time = []#outputs
    hiddens_time = []#hidden activations over time
    for input_item in inputs:  # Iterate over time steps.
        hidden_units = update_rnn(input_item, hidden_units)
        outputs_time.append(hidden_to_output_probs(hidden_units))
        if return_hiddens: hiddens_time.append(hidden_units)
    return np.array(outputs_time), jnp.array(hiddens_time)

def calc_performance_measures(outputs,targets):
    perform_measures = {}
    
    #Get accuracy
    # cutoff_start = 10
    classify_threshold_fixation = 0.15
    classify_threshold_decision = 0.1
    classify_period = -30
    total_num_predictions = np.size(targets,axis=1)
    num_output_vars = np.size(targets,axis=2)
    accuracy_peroutputvar = []
    accuracy_stdm_peroutput_var = []
    for output_var in range(num_output_vars):
        if output_var == 0:#Fixation signal needs to be maintained at all times
            targets_grandmean = np.mean(targets[classify_period:,:,output_var],axis=0)
            outputs_grandmean = np.mean(outputs[classify_period:,:,output_var],axis=0)
            correct_predictions = [ np.abs(i)-classify_threshold_fixation*np.abs(i) <= x < np.abs(i)+classify_threshold_fixation*np.abs(i) for x, i in zip(outputs_grandmean,targets_grandmean)]
        else:#onnly look at selection period at the end
            targets_grandmean = np.mean(targets[classify_period:,:,output_var],axis=0)
            outputs_grandmean = np.mean(outputs[classify_period:,:,output_var],axis=0)
            correct_predictions = [ np.abs(i)-classify_threshold_decision*np.abs(i) <= x < np.abs(i)+classify_threshold_decision*np.abs(i) for x, i in zip(outputs_grandmean,targets_grandmean)]
        
        accuracy_curr = (np.sum(correct_predictions)/total_num_predictions)*100
        accuracy_peroutputvar.append(accuracy_curr)
        accuracy_stdm_curr = np.std(correct_predictions)/np.sqrt(total_num_predictions)
        accuracy_stdm_peroutput_var.append(accuracy_stdm_curr)
        perform_measures[f"accuracy_OutputVariableNr{output_var+1}"] = accuracy_curr
        perform_measures[f"accuracy_stdm_OutputVariableNr{output_var+1}"] = accuracy_stdm_peroutput_var
           

    accuracy_peroutputvar = np.array(accuracy_peroutputvar)
    perform_measures['accuracy_overall'] = np.mean(accuracy_peroutputvar)
    perform_measures['accuracy_stdm_overall'] = np.mean(accuracy_stdm_peroutput_var)
    
    return perform_measures

def module_weight_silencer(results,data_params,module_to_silence):
    """
    Silence entire chunks of the network. module_to_silence parameter indicates what to do
    
    0: silence ALL inter-module weights
    1: silence ALL module 1 weights
    2: silence ALL module 2 weights
    ...

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights

    """
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    module_size = results['module_size'];
    input_size = results['input_size'];
    number_modules = results['number_modules']
    trained_network = results['trained_network']
    weights = trained_network['change'][input_size:,:]
    
    dims_rec_weights = np.shape(weights)
    silencing_mask_rec = np.ones((dims_rec_weights[0],dims_rec_weights[1]))
    # dims_predict_weights = np.shape(trained_network['predict'])
    # silencing_mask_out = np.zeros((dims_predict_weights[0],dims_predict_weights[1]))
    
    #Silence FIRST canoncial var module
    if module_to_silence == 0:#silence all inter-module weights
        silencing_mask_rec = np.zeros((dims_rec_weights[0],dims_rec_weights[1]))
        for module_num in range(1,number_modules+1):
            start_idx_x = module_size*(module_num-1)
            end_idx_x = module_size*module_num
            start_idx_y = module_size*(module_num-1)
            end_idx_y = module_size*module_num
            silencing_mask_rec[start_idx_x:end_idx_x,start_idx_y:end_idx_y] = 1
            
        weights = np.multiply(weights,silencing_mask_rec)
            
    elif module_to_silence == 1:
        silencing_mask_rec[:module_size,:module_size] = 0
        weights = np.multiply(weights,silencing_mask_rec)
        
    #Silence SECOND canoncial var module
    elif module_to_silence == 2:
        silencing_mask_rec[module_size:module_size*2,module_size:module_size*2] = 0
        weights = np.multiply(weights,silencing_mask_rec)
        
    #Silence THIRD canoncial var module
    elif module_to_silence == 3:
        silencing_mask_rec[-module_size:,-module_size:] = 0
        weights = np.multiply(weights,silencing_mask_rec)
        
    trained_network['change'][input_size:,:] = weights

    return trained_network

def percent_intermodule_weight_silencer(results,data_params,module_to_silence,percentage_to_lesion):
    """
    Silence percentages of module weights. module_to_silence parameter indicates which module to target
    
    1: silence PERCENTAGE of module 1 weights
    2: silence PERCENTAGE of module 2 weights
    ...
    
    Currently does not offer capability of silencing INTER-module weights

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights
        
        trained_network['change']: now is updated to silenced weight matrix
        trained_netwrok['perturbed_mask']: contains mask of which weights were silenced

    """
    trained_network_lesioned = percent_wholenetwork_weight_silencer(results,data_params,percentage_to_lesion)
    
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    module_size = results['module_size'];
    input_size = results['input_size'];
    number_modules = results['number_modules']
    trained_network = results['trained_network']
    copy_full_weights = trained_network['change'][input_size:,:]
    copy_full_weights_lesioned = trained_network_lesioned['change'][input_size:,:]
    
    shape_full_weights = np.shape(copy_full_weights)
    perturbed_mask = np.ones((shape_full_weights[0],shape_full_weights[1]))
    
    if percentage_to_lesion != 0:
        for module_num in range(1,number_modules+1):
            start_idx_x = module_size*(module_num-1)
            end_idx_x = module_size*module_num
            start_idx_y = module_size*(module_num-1)
            end_idx_y = module_size*module_num
            copy_full_weights_lesioned[start_idx_x:end_idx_x,start_idx_y:end_idx_y] = copy_full_weights[start_idx_x:end_idx_x,start_idx_y:end_idx_y]

    trained_network['change'][input_size:,:] = copy_full_weights_lesioned
    trained_network['perturbed_mask'] = np.array(perturbed_mask)
    
    return trained_network

def percent_module_weight_silencer(results,data_params,module_to_silence,percentage_to_lesion):
    """
    Silence percentages of module weights. module_to_silence parameter indicates which module to target
    
    1: silence PERCENTAGE of module 1 weights
    2: silence PERCENTAGE of module 2 weights
    ...
    
    Currently does not offer capability of silencing INTER-module weights

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights
        
        trained_network['change']: now is updated to silenced weight matrix
        trained_netwrok['perturbed_mask']: contains mask of which weights were silenced

    """
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    module_size = results['module_size'];
    input_size = results['input_size'];
    trained_network = results['trained_network']
    copy_full_weights = trained_network['change'][input_size:,:]
    
    shape_full_weights = np.shape(copy_full_weights)
    perturbed_mask = np.ones((shape_full_weights[0],shape_full_weights[1]))
    
    if percentage_to_lesion != 0:
        
        start_idx_x = module_size*(module_to_silence-1)
        end_idx_x = module_size*module_to_silence
        start_idx_y = module_size*(module_to_silence-1)
        end_idx_y = module_size*module_to_silence
        
        copy_module_weights = copy_full_weights[start_idx_x:end_idx_x,start_idx_y:end_idx_y]
        shape_module_weights = np.shape(copy_module_weights)
    
    
        #Get indices of weights to silence by random draw

        num_weights_total = shape_module_weights[0]*shape_module_weights[1]
        silenced_idx = np.random.choice(shape_module_weights[0]*shape_module_weights[1],
                                      np.int(np.floor((percentage_to_lesion/100)*num_weights_total)),replace=False,p=None)
        copy_module_weights = np.reshape(copy_module_weights,(num_weights_total,))
        copy_module_weights[silenced_idx] = 0
        # copy_module_weights[silenced_idx] = ops.index_update(copy_module_weights,ops.index[silenced_idx,],0)
        copy_module_weights = np.reshape(copy_module_weights,(shape_module_weights[0],shape_module_weights[1]))
        copy_full_weights[start_idx_x:end_idx_x,start_idx_y:end_idx_y] = copy_module_weights
        
        perturbed_mask_curr = np.ones((num_weights_total,))
        perturbed_mask_curr[silenced_idx] = 0
        perturbed_mask_curr = np.reshape(perturbed_mask_curr,(shape_module_weights[0],shape_module_weights[1]))
        perturbed_mask[start_idx_x:end_idx_x,start_idx_y:end_idx_y] = perturbed_mask_curr
        
        trained_network['change'][input_size:,:] = np.array(copy_full_weights)

    
    trained_network['perturbed_mask'] = np.array(perturbed_mask)
        
    
    return trained_network

def percent_wholenetwork_weight_silencer(results,data_params,percentage_to_lesion):
    """
    Silence percentages of ENTIRE NETWORK weights. 
    ...

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights
        
        trained_network['change']: now is updated to silenced weight matrix
        trained_netwrok['perturbed_mask']: contains mask of which weights were silenced

    """
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    input_size = results['input_size'];
    trained_network = results['trained_network']
    copy_full_weights = trained_network['change'][input_size:,:]
    
    shape_full_weights = np.shape(copy_full_weights)
    perturbed_mask = np.ones((shape_full_weights[0],shape_full_weights[1]))
    
    if percentage_to_lesion != 0:
        
        shape_module_weights = np.shape(copy_full_weights)
    
        #Get indices of weights to silence by random draw
        num_weights_total = shape_module_weights[0]*shape_module_weights[1]
        silenced_idx = np.random.choice(shape_module_weights[0]*shape_module_weights[1],
                                      np.int(np.floor((percentage_to_lesion/100)*num_weights_total)),replace=False,p=None)
        copy_full_weights = np.reshape(copy_full_weights,(num_weights_total,))
        copy_full_weights[silenced_idx] = 0
        copy_full_weights = np.reshape(copy_full_weights,(shape_module_weights[0],shape_module_weights[1]))

        perturbed_mask_curr = np.ones((num_weights_total,))
        perturbed_mask_curr[silenced_idx] = 0
        perturbed_mask_curr = np.reshape(perturbed_mask_curr,(shape_module_weights[0],shape_module_weights[1]))
        perturbed_mask[:,:] = perturbed_mask_curr
        
        trained_network['change'][input_size:,:] = np.array(copy_full_weights)

    
    trained_network['perturbed_mask'] = np.array(perturbed_mask)
    
    return trained_network

def percent_wholenetwork_activitylesion(results,data_params,percentage_to_lesion):
    """
    Silence percentages of ENTIRE NETWORK weights. 
    ...

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights
        
        trained_network['change']: now is updated to silenced weight matrix
        trained_netwrok['perturbed_mask']: contains mask of which weights were silenced

    """
    def rnn_predict_silence(params, inputs, silenced_mask, return_hiddens=False):
        def update_rnn(input_item, hidden_units):
            return jnp.tanh(concat_multiply(params['change'], input_item, hidden_units))
    
        def hidden_to_output_probs(hidden_units):
            return concat_multiply(params['predict'], hidden_units)
    
        batch_size = inputs.shape[1]
        hidden_units = jnp.repeat(params['hidden unit'], batch_size, axis=0)
        # print(np.shape(hidden_units))
        silencing_idx_batch = jnp.repeat(silenced_mask, batch_size, axis=0)
        # print(np.shape(silencing_idx_batch))
        outputs_time = []#outputs
        hiddens_time = []#hidden activations over time
        for input_item in inputs:  # Iterate over time steps.
            hidden_units = update_rnn(input_item, np.multiply(hidden_units,silencing_idx_batch))
            outputs_time.append(hidden_to_output_probs(hidden_units))
            if return_hiddens: hiddens_time.append(hidden_units)
        return np.array(outputs_time), np.array(hiddens_time)
    
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    trained_network = results['trained_network']
    
    shape_hiddens = np.shape(trained_network['hidden unit'])
    num_hiddens = shape_hiddens[0]*shape_hiddens[1]
    silenced_mask = np.ones((shape_hiddens[0],shape_hiddens[1]))
    perturbed_mask = np.ones((num_hiddens,))
    
    if percentage_to_lesion != 0:
    
        #Get indices of weights to silence by random draw
        silenced_idx = np.random.choice(num_hiddens,
                                      int(np.floor((percentage_to_lesion/100)*num_hiddens)),replace=False,p=None)
        silenced_mask = np.squeeze(silenced_mask)
        silenced_mask[silenced_idx] = 0
        silenced_mask = np.expand_dims(silenced_mask,axis=0)

        perturbed_mask = np.ones((num_hiddens,))
        perturbed_mask[silenced_idx] = 0
        perturbed_mask = np.expand_dims(perturbed_mask,axis=1)


    trained_network['perturbed_mask'] = np.array(perturbed_mask)
    
   
    outputs_net, hiddens_net = rnn_predict_silence(trained_network, results['test_inputs'],
                                                   silenced_mask, return_hiddens=True) 
    
    
    return outputs_net, hiddens_net

def percent_wholenetwork_addnoise(results,data_params,sigma_noise_toadd):
    """
    Add global noise to ENTIRE NETWORK weights. 
    ...

    Args:
        trained_network (dict): network weights.
        params (tuple): testing related params
        silece_params (tuple): silencing related params.

    Returns:
        trained_network (dict): containig network with silenced weights
        
        trained_network['change']: now is updated to silenced weight matrix
        trained_netwrok['perturbed_mask']: contains mask of which weights were silenced

    """
    batch_size_test,trial_num_plot,lesion_percent_plot,gsmooth_sigma,\
        rounding_precision,rand_key_idx_test,num_repetitions_rand_draws = data_params
    
    trained_network = results['trained_network']
    input_size = results['input_size']
    copy_full_weights = trained_network['change'][input_size:,:]
    shape_full_weights = np.shape(copy_full_weights)
    perturbed_mask = np.ones((shape_full_weights[0],shape_full_weights[1]))
    
    shape_weights = np.shape(copy_full_weights)
    global_noise_toadd = np.random.normal(0,sigma_noise_toadd,(shape_weights[0],shape_weights[1]))
    copy_full_weights = copy_full_weights + global_noise_toadd
    trained_network['change'][input_size:,:] = copy_full_weights
    
    perturbed_mask = perturbed_mask + global_noise_toadd
    trained_network['perturbed_mask'] = np.array(perturbed_mask)
    
    return trained_network


def rnn_mse(params, inputs, targets):
    outputs_time, _ = rnn_predict(params, inputs)
    return np.mean((outputs_time - targets)**2)

def rnn_rmse(params, inputs, targets):
    outputs_time, _ = rnn_predict(params, inputs)
    mse = np.mean((outputs_time - targets)**2)
    return np.sqrt(mse)

@optimizer
def adam_custom(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam.
  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).
  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = np.zeros_like(x0)
    v0 = np.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x_step, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * np.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i + 1))
    x_step = step_size(i) * mhat / (np.sqrt(vhat) + eps)
    return x_step, m, v
  def get_params(state):
    x_step, _, _ = state
    return x_step
  return init, update, get_params

def create_rnn_params(input_size, state_size, output_size,
                      g, random_key):

    input_factor = 1.0 / jnp.sqrt(input_size)
    hidden_scale = 1.0
    hidden_factor = g / jnp.sqrt(state_size+1)
    predict_factor = 1.0 / jnp.sqrt(state_size+1)
    return {'hidden unit': jnp.array(random.normal(random_key,(1, state_size)) * hidden_scale),
            'change':       jnp.concatenate((random.normal(random_key,(input_size, state_size)) * input_factor,
                                            random.normal(random_key,(state_size+1, state_size)) * hidden_factor),
                                           axis=0),#hidden weights
            'predict':      random.normal(random_key,(state_size+1, output_size)) * predict_factor}#readout weights

def create_rnn_params_symmetric(input_size, state_size, output_size,
                      g, random_key):

    input_factor = 1.0 / jnp.sqrt(input_size)
    hidden_scale = 1.0
    hidden_factor = g / jnp.sqrt(state_size+1)
    predict_factor = 1.0 / jnp.sqrt(state_size+1)
    normal_mat = random.normal(random_key,(state_size, state_size)) * hidden_factor
    return {'hidden unit': jnp.array(random.normal(random_key,(1, state_size)) * hidden_scale),
            'change':       jnp.concatenate((random.normal(random_key,(input_size, state_size)) * input_factor,
                                            jnp.tril(normal_mat)+np.transpose(np.tril(normal_mat,-1)),
                                            random.normal(random_key,(1,state_size))*hidden_factor),
                                           axis=0),#hidden weights
            'predict':      random.normal(random_key,(state_size+1, output_size)) * predict_factor}#readout weights
    

def rescale_activity(inputs,targets):
    #Rescale
    min_val_in = jnp.min(inputs.reshape(-1,)); 
    max_val_in = jnp.max(inputs.reshape(-1,));
    inputs_rescaled = jnp.array([(x-min_val_in) / (max_val_in - min_val_in) for x in inputs])
    
    min_val_out = jnp.min(targets.reshape(-1,)); 
    max_val_out = jnp.max(targets.reshape(-1,));
    targets_rescaled = jnp.array([(x-min_val_out) / (max_val_out - min_val_out) for x in targets])
    return inputs_rescaled,targets_rescaled

############## MODULAR TRAINING DUNCKER TASKS ##################
def train_weightfreeze_modular_aggregate_dynamics(train_params,task_params, leave_task_out=None,init_params=None):
    
    state_size,module_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,\
    init_step_size, weightfreeze_mask_dict, fitted_dimreduction_model, optimizer_b1,\
        optimizer_b2,optimizer_eps,pc_select, task_select, rand_key_idx_task,\
            save_dir, save_name, dropout_on, decrease_lr_on, train_flag = train_params
            
    
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                    dim_reduction_flag, do_plot, do_save = task_params
                    
    rand_gen_counter = rand_key_idx_task#Set start for random seed
        
    # Allow continued training.
    if init_params is None:
        init_params = create_rnn_params(input_size=1, output_size=1,
                                        state_size=state_size, 
                                        g=g)
    #Get training data loss for incremental task training
    def training_loss(params, iter):
        
        #Set random seed start to rand_gen_counter and get new task params
        task_params = rand_gen_counter, task_dict, batch_size,bval, sval, T, ntime,\
            n_components_pca,dim_reduction_flag,do_plot, do_save
                
        if task_select == 'dynamics_pretrain':
            #Draw fresh inputs, targets
            inputs, targets = aggregate_tasks(task_params,leave_task_out) 
            
            # inputs_shape = jnp.shape(inputs)
            # targets_shape = jnp.shape(targets)

            
            if dim_reduction_flag == 'PCA':
                #Performs live projections
                fitted_dimreduction_model = get_pca_components(inputs,targets,task_params)
                inputs = fitted_dimreduction_model['inputs_pca']
                targets = fitted_dimreduction_model['targets_pca']
                
            elif dim_reduction_flag == 'CANCOR':
                # inputs, targets = transform_data_dimreduce(inputs,targets,1)
                # inputs,targets = fitted_dimreduction_model['cca_fit_model'].transform(inputs, targets)
                # inputs, targets = transform_data_dimreduce(inputs,targets,-1)
                
                #Performs live projections
                fitted_dimreduction_model = get_cancor_components(inputs,targets,task_params)
                inputs = fitted_dimreduction_model['inputs_cca']
                targets = fitted_dimreduction_model['targets_cca']
                
            #Subselect PCs for training
            inputs = inputs[:,:,pc_select-1:pc_select]
            targets = targets[:,:,pc_select-1:pc_select]
                
            
        elif task_select == 'delay_pro':
            inputs,targets = build_delaypro_task(task_params)
        elif task_select == 'delay_anti':
            inputs,targets = build_delayanti_task(task_params)
        elif task_select == 'mem_pro':
            inputs,targets = build_mempro_task(task_params)
        elif task_select == 'mem_anti':
            inputs,targets = build_memanti_task(task_params)
        elif task_select == 'mem_dm1':
            inputs,targets = build_memdm1_task(task_params)
        elif task_select == 'mem_dm2':
            inputs,targets = build_memdm2_task(task_params)
        elif task_select == 'context_mem_dm1':
            inputs,targets = build_contextmemdm1_task(task_params)
        elif task_select == 'context_mem_dm2':
            inputs,targets = build_contextmemdm2_task(task_params)
        elif task_select == 'multi_mem':
            inputs,targets = build_multimem_task(task_params)
            
            
        #Rescale
        inputs,targets = rescale_activity(inputs,targets)

        mse = rnn_mse(params, inputs, targets)
        l2_reg = 2e-6
        reg_loss = l2_reg * jnp.sum(params['change']**2)
                      
        # mse = rnn_mse(params, inputs, targets)
        # if dropout_on:
        #     #Dropout for recurrent weights
        #     # random_key = random.PRNGKey(rand_gen_counter)
        #     # normal_mask = 1+random.normal(random_key,(state_size+input_size+1,state_size))
        #     # params_copy = params
        #     # params_copy['change'] = np.multiply(params_copy['change'],normal_mask)
        
        return mse + reg_loss        

              
    #Get optimizer & define update step
    if decrease_lr_on:
        step_sizes = iter([init_step_size*(0.333333**n) for n in range(nstep_sizes)])
    else:
        step_sizes = iter([init_step_size for n in range(nstep_sizes)])
        
    opt_init, opt_update, get_params = adam(step_size=next(step_sizes),
                                            b1=optimizer_b1, b2=optimizer_b2, eps=optimizer_eps)
    trained_params = init_params
    opt_state = opt_init(init_params) 
    
    def weightmask_multiply(g):   
        # print(f"Recurrent weight mask:{weightfreeze_mask_dict['change']}")
        # g['change']  = ops.index_update(g['change'] ,ops.index[:,:],
        #                                 jnp.multiply(g['change'],weightfreeze_mask_dict['change']))
        # g['predict']  = ops.index_update(g['predict'] ,ops.index[:,:],
        #                                  jnp.multiply(g['predict'],weightfreeze_mask_dict['predict']))
        g['change']  = jnp.multiply(g['change'],weightfreeze_mask_dict['change'])
        g['predict']  = jnp.multiply(g['predict'],weightfreeze_mask_dict['predict'])
        # if dropout_on:
            #Dropout for recurrent weights
            # random_key = random.PRNGKey(rand_gen_counter)
            # normal_mask = 1+random.normal(random_key,(state_size+input_size+1,state_size))
            # g['change'] = np.multiply(g['change'],normal_mask)
            
            # #Dropout for output weights
            # random_key = random.PRNGKey(rand_gen_counter)
            # normal_mask = 1+random.normal(random_key,(np.size(g['predict'],axis=0),
            #                                           np.size(g['predict'],axis=1)))
            # g['predict'] = np.multiply(g['predict'],normal_mask)
        return g  
    
    @jit
    def step(i,opt_state,batch):
        params = get_params(opt_state)
        # params_updated = weightmask_multiply(params)
        g = grad(training_loss)(params,i)#build gradient of loss function using jax grad
        g = weightmask_multiply(g)
        return opt_update(i, g, opt_state)
    
    loss_log = []
    if train_flag != 0:
        for i,step_size in enumerate(step_sizes):
            print("(%d/%d) Training RNN for %d steps at step size %f" % (i+1, nstep_sizes, num_iters, step_size))
            for j in range(num_iters):
                # if train_flag == 0: print('NO TRAINING'); continue
                opt_state = step(i,opt_state,trained_params)#Run optimization step
                rand_gen_counter = rand_gen_counter + 1
                if j % 10 == 0:#Print out current loss
                    curr_loss = training_loss(get_params(opt_state), 0)
                    loss_log.append(curr_loss)
                    print("Iteration: ", j, " Train loss:", curr_loss)
             
            #Get temp results at step_size change & save    
            trained_params = get_params(opt_state)
            
            #Save trained params at each step size change
            filehandler = open(save_dir+"/RNN_trained_params_"+save_name+'.pickle',"wb")
            pickle.dump(trained_params,filehandler)
            pickle.dump(train_params,filehandler)
            pickle.dump(np.array(loss_log), filehandler)
            filehandler.close()   
    else:
        print('NO TRAINING')
        #Save trained params at each step size change
        filehandler = open(save_dir+"/RNN_trained_params_"+save_name+'.pickle',"wb")
        pickle.dump(trained_params,filehandler)
        pickle.dump(train_params,filehandler)
        pickle.dump(np.array(loss_log), filehandler)
        filehandler.close()   

    print("Done.")
    return trained_params

def check_input_output_dims(inputs,targets,input_size,output_size):
    #If less inputs than expected from previous tasks, concatenate with zero vecs to get same number
    input_size_curr = np.size(inputs,axis=2)
    target_size_curr = np.size(targets,axis=2)
    
    if input_size != input_size_curr or output_size != target_size_curr:

        if input_size_curr < input_size :
            inputs = np.concatenate((inputs,np.zeros((np.size(inputs,axis=0),
                                                      np.size(inputs,axis=1),
                                                      input_size - input_size_curr))),axis=2)
        if target_size_curr < output_size :
            targets = np.concatenate((targets,np.zeros((np.size(targets,axis=0),
                                                      np.size(targets,axis=1),
                                                      output_size - target_size_curr))),axis=2) 
        if input_size_curr > input_size:
            inputs=inputs[:,:,:input_size]
        if target_size_curr > output_size:
            targets=targets[:,:,:output_size]
    return inputs,targets  

def train_DNI_method(train_params,task_params, init_params=None, cov_mat_prev=None, projection_vars_prev=None,compute_loss_alltasks_flag=False):
    '''
    Updated Dunckertask with functions function from Duncker code:
        - compute_projection_matrices
        - rescale_cov_evals
        - compute_covariance
    Gets rid of high var dims
    '''
    state_size,module_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,\
            init_step_size, weightfreeze_mask_dict,l2_reg_val,a_reg_val,task_select, rand_key_idx_task,\
            l2_reg_val,a_reg_val,save_dir, save_name, dropout_on, decrease_lr_on, train_flag,\
                covmat_update_task_counter= train_params
            
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                    dim_reduction_flag, do_plot, do_save = task_params

    rand_gen_counter = rand_key_idx_task#Set start for random seed
    
    duncker_tasks = ['delay_pro','delay_anti','mem_pro','mem_anti',\
                    'mem_dm1','mem_dm2','context_mem_dm1','context_mem_dm2','multi_mem']
        
    # Allow continued training.
    if init_params is None:
        init_params = create_rnn_params(input_size=2, output_size=1,
                                        state_size=state_size, 
                                        g=g)
    
    def calc_mse(outputs,targets):
        return jnp.mean((outputs - targets)**2)  
    
    def build_task_helper(task_select,task_params):
        if task_select == 'delay_pro':
            inputs,targets = build_delaypro_task(task_params)
        elif task_select == 'delay_anti':
            inputs,targets = build_delayanti_task(task_params)
        elif task_select == 'mem_pro':
            inputs,targets = build_mempro_task(task_params)
        elif task_select == 'mem_anti':
            inputs,targets = build_memanti_task(task_params)
        elif task_select == 'mem_dm1':
            inputs,targets = build_memdm1_task(task_params)
        elif task_select == 'mem_dm2':
            inputs,targets = build_memdm2_task(task_params)
        elif task_select == 'context_mem_dm1':
            inputs,targets = build_contextmemdm1_task(task_params)
        elif task_select == 'context_mem_dm2':
            inputs,targets = build_contextmemdm2_task(task_params)
        elif task_select == 'multi_mem':
            inputs,targets = build_multimem_task(task_params)
        return inputs,targets
    
    def compute_projection_matrices(Swh, Suh, Syy, Shh, alpha):
        # ------ rescaled eigenvalue approach ------
    
        # get eigendecomposition of covariance matrices
        Dwh, Vwh = jnp.linalg.eig(Swh)
        Duh, Vuh = jnp.linalg.eig(Suh)
        Dyy, Vyy = jnp.linalg.eig(Syy)
        Dhh, Vhh = jnp.linalg.eig(Shh)
    
        # recompute eigenvalue
        Dwh_scaled = rescale_cov_evals(Dwh, alpha)
        Duh_scaled = rescale_cov_evals(Duh, alpha)
        Dyy_scaled = rescale_cov_evals(Dyy, alpha)
        Dhh_scaled = rescale_cov_evals(Dhh, alpha)
    
        # reconstruct projection matrices with eigenvalues rescaled (and inverted: high variance dims are zero-ed out)
        P1 = jnp.matmul(jnp.matmul(Vwh, jnp.diag(Dwh_scaled)), jnp.transpose(Vwh)) # output space W cov(Z) W'
        P2 = jnp.matmul(jnp.matmul(Vuh, jnp.diag(Duh_scaled)), jnp.transpose(Vuh))  # input space cov(Z)
        P3 = jnp.matmul(jnp.matmul(Vyy, jnp.diag(Dyy_scaled)), jnp.transpose(Vyy))  # readiyt space  cov(Y)
        P4 = jnp.matmul(jnp.matmul(Vhh, jnp.diag(Dhh_scaled)), jnp.transpose(Vhh))  # recurrent space cov(H)

        return jnp.real(P1), jnp.real(P2), jnp.real(P3), jnp.real(P4)
    
    
    def rescale_cov_evals(evals, alpha):
        # ---- cut-off ----
        fvals = alpha / (alpha + evals)
    
        return fvals
    
    
    def compute_covariance(x):
        # computes X * X.T
        return jnp.matmul(x, jnp.transpose(x)) / (jnp.shape(x)[1] - 1)  # or use biased estimate?
        
        
    def rnn_predict_duncker(params, inputs, state_size, return_hiddens=True):
        """
        Implements Duncker & Driscoll method NeurIPS 2020
        """    
        
        def compute_projections(params, hiddens_and_inputs_time,hiddens_time,outputs_time):
    
            alpha = 0.001 #small constant to ensure regularisation & invertibility
                
            #Input&recurrent cov mat
            shape_handi = jnp.shape(hiddens_and_inputs_time)
            hiddens_and_inputs_time_r = jnp.reshape(hiddens_and_inputs_time,\
                                                   (shape_handi[0]*shape_handi[1],shape_handi[2]))
            hiddens_and_inputs_time_r = jnp.moveaxis(hiddens_and_inputs_time_r,0,1)
            
            ih_cov_mat = compute_covariance(hiddens_and_inputs_time_r)
            
            #Activity covariance matrix 
            wa_cov_mat = jnp.matmul(jnp.matmul(jnp.transpose(params['change']),ih_cov_mat),params['change'])
            
            #Recurrent covariance matrix
            shape_handi = jnp.shape(hiddens_time)
            hiddens_time_r = jnp.reshape(hiddens_time,(shape_handi[0]*shape_handi[1],shape_handi[2]))
            hiddens_time_r = jnp.moveaxis(hiddens_time_r,0,1)
            hiddens_time_r = jnp.concatenate((hiddens_time_r,jnp.ones((1,jnp.size(hiddens_time_r,axis=1)))),axis=0)
            h_cov_mat = compute_covariance(hiddens_time_r)
            
            #Output covariance matrix
            shape_o = jnp.shape(outputs_time)
            outputs_time_r = jnp.reshape(outputs_time,(shape_o[0]*shape_o[1],shape_o[2]))
            outputs_time_r = jnp.moveaxis(outputs_time_r,0,1)
            o_cov_mat = compute_covariance(outputs_time_r)

            #If there are cov_mats from previous tasks, combine & update
            if cov_mat_prev is not None: 
                ih_cov_mat_prev = cov_mat_prev[0]
                #Add previous task's covariance matrix
                ih_cov_mat = jnp.divide((covmat_update_task_counter-1),covmat_update_task_counter)*ih_cov_mat_prev +\
                    jnp.divide(1,covmat_update_task_counter)*ih_cov_mat
                    
                wa_cov_mat_prev = cov_mat_prev[1]
                #Add previous task's covariance matrix
                wa_cov_mat = jnp.divide((covmat_update_task_counter-1),covmat_update_task_counter)*wa_cov_mat_prev +\
                    jnp.divide(1,covmat_update_task_counter)*wa_cov_mat
                    
                h_cov_mat_prev = cov_mat_prev[2]
                #Add previous task's covariance matrix
                h_cov_mat = jnp.divide((covmat_update_task_counter-1),covmat_update_task_counter)*h_cov_mat_prev +\
                    jnp.divide(1,covmat_update_task_counter)*h_cov_mat
                    
                o_cov_mat_prev = cov_mat_prev[3]
                #Add previous task's covariance matrix
                o_cov_mat = jnp.divide((covmat_update_task_counter-1),covmat_update_task_counter)*o_cov_mat_prev +\
                    jnp.divide(1,covmat_update_task_counter)*o_cov_mat
            
            #Compute projection vars with updated cov mats
            p_ih = jnp.linalg.inv((1/alpha)*ih_cov_mat+jnp.identity(jnp.shape(ih_cov_mat)[0]))
            p_wa = jnp.linalg.inv((1/alpha)*wa_cov_mat + jnp.identity(jnp.shape(params['change'])[1]))
            
            #Compute recurrent & output projection vars
            p_h = jnp.linalg.inv((1/alpha)*h_cov_mat + jnp.identity(jnp.shape(h_cov_mat)[0]))
            p_o = jnp.linalg.inv((1/alpha)*o_cov_mat + jnp.identity(jnp.shape(params['predict'])[1]))
            
            # p_wa,p_ih,p_o,p_h = compute_projection_matrices(wa_cov_mat, ih_cov_mat, o_cov_mat, h_cov_mat, jnp.int_(alpha))
            
            return (ih_cov_mat, wa_cov_mat,h_cov_mat,o_cov_mat), (p_ih, p_wa, p_h, p_o)
        
        def update_rnn(input_item, hidden_units):
            return jnp.tanh(concat_multiply(params['change'], input_item, hidden_units))
    
        def hidden_to_output_probs(hidden_units):
            return concat_multiply(params['predict'], hidden_units)
        
    
        batch_size_curr = inputs.shape[1]
        hidden_units = jnp.repeat(params['hidden unit'], batch_size_curr, axis=0)
        outputs_time = []#outputs
        hiddens_time = []#hidden activations over time
        hiddens_and_inputs_time = []#accumulate input & hidden projections over time
        for input_item in inputs:  # Iterate over time steps.
            hiddens_and_inputs_time.append(concat_args(input_item,hidden_units))
            hiddens_time.append(hidden_units)
            hidden_units = update_rnn(input_item, hidden_units)
            outputs_time.append(hidden_to_output_probs(hidden_units))
            
        #Compute projections
        cov_mats, projection_vars_local = compute_projections(params,jnp.array(hiddens_and_inputs_time),
                                                              jnp.array(hiddens_time),jnp.array(outputs_time))
        
        return jnp.array(outputs_time), jnp.array(hiddens_time), cov_mats, projection_vars_local
        
        
    def generate_projections(params,task_select,return_hiddens = False):
        inputs,targets = build_task_helper(task_select,task_params)
        inputs,targets = rescale_activity(inputs,targets)
        inputs,targets = check_input_output_dims(inputs,targets,input_size,output_size)
        outputs, hiddens_time, cov_mats, projection_vars_local = rnn_predict_duncker(params, inputs, state_size,return_hiddens)
        return inputs,targets,outputs,hiddens_time,cov_mats,projection_vars_local
    
    def compute_loss_alltasks(params,rand_gen_counter):
        task_params = rand_gen_counter, task_dict, batch_size,bval, sval, T, ntime,\
            n_components_pca,dim_reduction_flag,do_plot, do_save
        all_task_losses = []
        for task_select in duncker_tasks:
            inputs,targets = build_task_helper(task_select,task_params)
            inputs,targets = rescale_activity(inputs,targets)
            inputs,targets = check_input_output_dims(inputs,targets,input_size,output_size) 
            outputs,_ = rnn_predict(params, inputs, return_hiddens=False)
            loss_current_task = calc_mse(outputs,targets)
            all_task_losses.append(loss_current_task)
        return all_task_losses
        
    
    #Get training data loss for incremental task training
    def training_loss(params, iter):
        
        # #Update prediction vars in this step too so we can use them to update gradient later
        # global prediction_vars
        # prediction_vars = prediction_vars_curr
        # mse = calc_mse(outputs,targets)
        task_params = rand_gen_counter, task_dict, batch_size,bval, sval, T, ntime,\
            n_components_pca,dim_reduction_flag,do_plot, do_save
            
        inputs,targets = build_task_helper(task_select,task_params)
        inputs,targets = rescale_activity(inputs,targets)
        
        inputs,targets = check_input_output_dims(inputs,targets,input_size,output_size)         
            
        outputs_time, _ = rnn_predict(params, inputs)
        mse = jnp.mean((outputs_time - targets)**2)
    
        # Add L2 Weight Regularization
        l2_reg = l2_reg_val
        reg_loss = l2_reg * jnp.sum(params['change']**2)
        
        #Activity regularization
        a_reg = a_reg_val
        a_loss = a_reg * jnp.sum(params['hidden unit']**2)
        
        total_loss = mse + reg_loss + a_loss
        
        return total_loss
    
    def project_p(g, projection_vars):

        p_ih, p_wa, p_h, p_o= projection_vars
        
        # g['change'] = ops.index_update(g['change'],ops.index[:,:],
        #                         jnp.matmul(jnp.matmul(p_ih,g['change']),p_wa))
        # g['predict'] = ops.index_update(g['predict'],ops.index[:,:],
        #                         jnp.matmul(jnp.matmul(p_h,g['predict']),p_o))
        g['change'] = jnp.matmul(jnp.matmul(p_ih,g['change']),p_wa)
        g['predict'] = jnp.matmul(jnp.matmul(p_h,g['predict']),p_o)
        return g

              
    #Get optimizer & define update step
    step_sizes = iter([init_step_size*(0.333333**n) for n in range(nstep_sizes)])
    opt_init, opt_update, get_params = adam(step_size=next(step_sizes),
                                            b1=0.9, b2=0.999, eps=1e-8)
    trained_params = init_params
    opt_state = opt_init(init_params) 
    
    #Initialize global prediction vars
    if cov_mat_prev is None:
        #initialize projection vars to identity
        _,_,_,_,_,projection_vars = generate_projections(trained_params,task_select)
        p_ih, p_wa, p_h, p_o = projection_vars
        
        p_ih_i = jnp.identity(jnp.shape(p_ih)[0])
        p_wa_i = jnp.identity(jnp.shape(p_wa)[0])
        p_h_i = jnp.identity(jnp.shape(p_h)[0])
        p_o_i = jnp.identity(jnp.shape(p_o)[0])
        projection_vars = p_ih_i, p_wa_i, p_h_i, p_o_i
    else:
        #Get cov_mat & projection vars calculated on last step of previous task
        projection_vars = projection_vars_prev
        # cov_mat = cov_mat_prev
        
    
    @jit
    def step(i,opt_state,projection_vars):
        params = get_params(opt_state)
        g = grad(training_loss)(params,i)#build gradient of loss function using jax grad
        g_projected = project_p(g, projection_vars)
        return opt_update(i, g_projected, opt_state)
    
    loss_log = []; loss_log_alltasks = []
    for i,step_size in enumerate(step_sizes):
        print("(%d/%d) Training RNN for %d steps at step size %f" % (i+1, nstep_sizes, num_iters, step_size))
        for j in range(num_iters):
            opt_state = step(i,opt_state,projection_vars)#Run optimization step
            rand_gen_counter = rand_gen_counter + 1
            if i==0 and j==0:
                #Get temp results at step_size change & save    
                trained_params = get_params(opt_state)
                loss_log_array = jnp.array(loss_log)
                _,_,_,_,cov_mats_save,projection_vars_save = generate_projections(trained_params,task_select)
                
                #Save trained params at each step size change
                filehandler = open(save_dir+"/RNN_trained_params_"+save_name+'.pickle',"wb")
                pickle.dump(trained_params,filehandler)
                pickle.dump(loss_log_array, filehandler)
                pickle.dump(cov_mats_save, filehandler)
                pickle.dump(projection_vars_save, filehandler)
                pickle.dump(loss_log_alltasks, filehandler)
                filehandler.close() 
                
            if j % 10 == 0:#Print out current loss
                curr_loss = training_loss(get_params(opt_state), 0)
                loss_log.append(curr_loss)
                if compute_loss_alltasks_flag: 
                    curr_loss_alltasks = compute_loss_alltasks(get_params(opt_state),rand_gen_counter)
                    loss_log_alltasks.append(curr_loss_alltasks)
                print("Iteration: ", j, " Train loss:", curr_loss)
         
    #Compute new projections after training 
    trained_params = get_params(opt_state)
    loss_log_array = jnp.array(loss_log)
    _,_,_,_,cov_mats_save,projection_vars_save = generate_projections(trained_params,task_select)
    
    #Save trained params at each step size change
    filehandler = open(save_dir+"/RNN_trained_params_"+save_name+'.pickle',"wb")
    pickle.dump(trained_params,filehandler)
    pickle.dump(cov_mats_save, filehandler)
    pickle.dump(projection_vars_save, filehandler)
    pickle.dump(loss_log_array, filehandler)
    pickle.dump(loss_log_alltasks, filehandler)
    filehandler.close()        

    print("Done.")
    return trained_params


