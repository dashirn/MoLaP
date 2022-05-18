#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 15:22:11 2022

@author: root
"""

import pickle
import jax.numpy as np
from jax import random, ops #https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers
from jax.config import config
config.update("jax_enable_x64", True)

import numpy as onp
# from jax.ops import index_update as jupdateidx
# import mlflow
# import mlflow.sklearn
# import logging

from rnn_build import create_rnn_params, train_weightfreeze_modular_aggregate_dynamics,train_DNI_method
from rnn_tasks import aggregate_tasks, get_pca_components, get_cancor_components, transform_data_dimreduce


# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

#### General parameters/directories
save_dir_testexamples = "/Users/davidavinci/Documents/MountSinai/Code/ContextSwitch_Weightfreeze_Modular"

#Directory where to save results
save_dir = "/Users/davidavinci/Documents/MountSinai/Code/modular_multitask_nets/results/tmp_train_22"

#Task dict: (input_size,output_size)
task_dict = {}
task_dict['delay_pro'] = task_dict['delay_anti'] = task_dict['mem_pro'] = task_dict['mem_anti'] = (3,3);
task_dict['mem_dm1'] = task_dict['mem_dm2'] = (2,2)
task_dict['context_mem_dm1'] = task_dict['context_mem_dm2'] = task_dict['multi_mem'] = (3,2)



def cancor_task_info(task_params,save_info):
    """
    Aggregates tasks one by one and checks how good reconstruction is from 
    using 1+ canonical variables.
    
    Fit one cancor model to a bag of tasks, leaving one out. Also fit another
    cancor model on the left out task. Then get vectors used to project into 
    CCA space (ca.x_rotations_) from both CCA models and get dot product for 
    between them for every dimension => Check how well the already trained components
    will generalize to new task/decide if CCA model needs to be re-trained or if
    another component needs to be added.

    Args:
        task_params (tuple): Task-related parameter settiungs.

    Returns:
        None.

    """
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                    dim_reduction_flag, do_plot, do_save = task_params
    save_dir,save_name = save_info
    
    inputs_stacked, targets_stacked = aggregate_tasks(task_params)
    original_inputs_shape = np.shape(inputs_stacked)
    original_targets_shape = np.shape(targets_stacked)
    original_shape_info = original_inputs_shape, original_targets_shape
    task_input_indices = range(25)
    task_output_indices = range(22)

    
    #Fit full model
    full_ccamodel = get_cancor_components(inputs_stacked,targets_stacked,task_params) 
    inputs_transformed, targets_transformed = transform_data_dimreduce(inputs_stacked,targets_stacked,1,original_shape_info)
    r2_score_full =  full_ccamodel['cca_fit_model'].score(inputs_transformed,targets_transformed)
    
    results = {}
    curr_input_distances = []
    curr_target_distances = []
    r2_scores_leftout = []
    input_size_idx = output_size_idx = 0
    task_counter = 1
    
    for task_select in task_dict:
        print(f"Cancor info | {task_select}")
        input_size_curr,output_size_curr = task_dict[task_select]
        input_size_idx += input_size_curr
        output_size_idx += output_size_curr
        leftout_task_input_indices = [i for i in range(input_size_idx-input_size_curr,input_size_idx)]
        leftout_task_target_indices = [i for i in range(output_size_idx-output_size_curr,output_size_idx)]
        current_bagoftasks_input_idx = [x for i,x in enumerate(task_input_indices) if i not in set(leftout_task_input_indices)]
        current_bagoftasks_target_idx = [x for i,x in enumerate(task_output_indices) if i not in set(leftout_task_target_indices)]

        #Fit model on bag of tasks minus left out task
        current_bagoftasks_ccamodel = get_cancor_components(inputs_stacked[:,:,current_bagoftasks_input_idx],
                                                            targets_stacked[:,:,current_bagoftasks_target_idx],
                                                            task_params)

        # #Fit model on left out task
        # current_leftouttask_ccamodel = get_cancor_components(inputs_stacked[:,:,leftout_task_input_indices],
        #                                                      targets_stacked[:,:,leftout_task_target_indices],
        #                                                      task_params_curr)
        
        
        #Compare new model to previous model via Euclidean distance
        input_distance = onp.linalg.norm(onp.squeeze(current_bagoftasks_ccamodel['inputs_cca']) - onp.squeeze(full_ccamodel['inputs_cca']))
        target_distance = onp.linalg.norm(onp.squeeze(current_bagoftasks_ccamodel['targets_cca']) - onp.squeeze(full_ccamodel['targets_cca']))
        curr_input_distances.append(input_distance)
        curr_target_distances.append(target_distance)
        
        #Get score for current model 
        curr_inputs_shape = np.shape(inputs_stacked[:,:,current_bagoftasks_input_idx])
        curr_targets_shape = np.shape(targets_stacked[:,:,current_bagoftasks_target_idx])
        curr_shape_info = curr_inputs_shape, curr_targets_shape
        inputs_transformed, targets_transformed = transform_data_dimreduce(inputs_stacked[:,:,current_bagoftasks_input_idx],
                                                                           targets_stacked[:,:,current_bagoftasks_target_idx],1,curr_shape_info)
        curr_score =  current_bagoftasks_ccamodel['cca_fit_model'].score(inputs_transformed,targets_transformed)
        r2_scores_leftout.append(curr_score)
        
        task_counter += 1
        
    results["input_distances"] = onp.array(curr_input_distances)
    results["target_distances"] = onp.array(curr_target_distances)
    results["coeffdetermination_full_model"] = r2_score_full
    results["coeffdetermination_leave_out_models"] = onp.array(r2_scores_leftout)
    results['task_params'] = task_params
        
    filehandler = open(save_dir+"/"+save_name+'.pickle',"wb")
    pickle.dump(results,filehandler) 
    filehandler.close()
    
    return results
            
        
    
def module_pretrain_pcdynamics(task_params,network_params,module_num_totrain,save_name_addon,leave_out_tasks=()):
    """
    Pretrains network modules on PC task dynamics.
    Module 1 on PC1, module 2 on PC2, etc.    
    
    For cancor: module 1 is trained on input pc 1 and target pc 1, etc.
    
    To see if pre-trained network can generalize to new task:
    Get CCA vars from a model trained on new task and pre-train a new
    network on those. Then correlate the test network activations for the trained
    modules of the old network (trained on bag of tasks) with new network (trained on new task)
    
    Args:
        task_params (tuple): Task-related parameter settiungs.
        network_params (tuple): Network related parameter settings.
        module_num_totrain (int): which module to train (1,2,3,...)

    Returns:
        None.

    """
    
    #Get parameters
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                    dim_reduction_flag, do_plot, do_save = task_params
    output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
        num_iters,nstep_sizes,init_step_size,\
            rand_key_network_init_params, save_dir, train_flag,\
                zero_out_intermoduleweights = network_params
    pc_select = module_num_totrain #module number is equivalent to pc number
    task_select = 'dynamics_pretrain'
    leave_out_tasks = set(leave_out_tasks)
    
    if train_flag == 0:
        save_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{module_num_totrain}_NoTrainControl_{save_name_addon}"
    else:
        save_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{module_num_totrain}_{save_name_addon}"
                    
    #Create initialization parameters
    if module_num_totrain == 1:
        init_params = create_rnn_params(input_size=input_size, output_size=output_size,
                                        state_size=state_size, g=g, random_key=rand_key_network_init_params)
        inputs_stacked, targets_stacked = aggregate_tasks(task_params,leave_out_tasks)
        
        if dim_reduction_flag == 'PCA':
            fitted_dimreduction_model = get_pca_components(inputs_stacked,targets_stacked,task_params)
        elif dim_reduction_flag == 'CANCOR':
            fitted_dimreduction_model = get_cancor_components(inputs_stacked,targets_stacked,task_params)
    else:
        
        if train_flag == 0:
            load_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{module_num_totrain-1}_NoTrainControl_{save_name_addon}"
        else:
            load_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{module_num_totrain-1}_{save_name_addon}"
        file_loc_trained = save_dir+"/RNN_trained_params_"+load_name+".pickle"
        file_loc = open(file_loc_trained,'rb')
        init_params = pickle.load(file_loc)
        train_params = pickle.load(file_loc)
        fitted_dimreduction_model = train_params[10]#get fitted pca models
    
    #SELECT WEIGHTS TO BE TRAINED 
    #Initialize recurrent weight matrix (J)
    weightfreeze_mask_recur = onp.concatenate((onp.zeros((input_size, state_size)),
                                            onp.zeros((state_size+1, state_size))),
                                            axis=0)
    weightfreeze_mask_dict = {}
    weightfreeze_mask_dict['hidden unit'] = np.ones((np.shape(init_params['hidden unit'])[0],
                                                     np.shape(init_params['hidden unit'])[1]))  
    
    #Input Weights PRE-Train ON.
    #Trains each pc input to its module only (not cross module weights)
    # weightfreeze_mask_recur[:input_size,
    #                         int(module_size*(module_num_totrain-1)):int(module_size*module_num_totrain)] = 1;
    # weightfreeze_mask_recur[:input_size,:] = 1;
    #Train all input weights (intra- and inter-module weights)
    # weightfreeze_mask_recur[int(pc_select-1):int(pc_select),
    #                         int(module_size*(module_num_totrain-1)):int(module_size*module_num_totrain)] = 1;
    weightfreeze_mask_recur[int(pc_select-1):int(pc_select),:] = 1;
    # weightfreeze_mask_recur[:n_components_pca,:] = 1;
    
    #Module weights PRE-Train ON
    index_start_row = int(input_size+module_size*(module_num_totrain-1))
    index_end_row = int(input_size+module_size*module_num_totrain)
    index_start_col = int(module_size*(module_num_totrain-1))
    index_end_col = int(module_size*module_num_totrain)
    weightfreeze_mask_recur[index_start_row:index_end_row,\
                                index_start_col:index_end_col] = 1;
    weightfreeze_mask_dict['change'] = np.array(weightfreeze_mask_recur)
    
    #Zero-out inter-module weights
    if zero_out_intermoduleweights:
        init_params['change'] =  ops.index_update(init_params['change'] ,ops.index[:,:],
                                         np.multiply(init_params['change'],weightfreeze_mask_dict['change']))
     # print(f"Shape weights:{np.shape(weightfreeze_mask_dict['change'])}")
    
    #Predict Weights Train ON
    weightfreeze_mask_dict['predict'] = np.ones((np.shape(init_params['predict'])[0],
                                                      np.shape(init_params['predict'])[1]))
    
    train_params = (state_size,module_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,
                    init_step_size, weightfreeze_mask_dict, fitted_dimreduction_model,
                    optimizer_b1, optimizer_b2,optimizer_eps,pc_select,task_select,
                    rand_key_idx_network_init_params, save_dir, save_name,
                    dropout_on,decrease_lr_on,train_flag)
    
    #Save initial parameter settings using pickle (faster)
    filehandler = open(save_dir+"/RNN_init_params_"+save_name+'.pickle',"wb")
    pickle.dump(init_params,filehandler) 
    pickle.dump(task_params,filehandler)
    filehandler.close()
    
    #Train
    print(f'PRE-TRAIN. Module {module_num_totrain}')
    train_weightfreeze_modular_aggregate_dynamics(train_params=train_params,task_params=task_params,
                                                  leave_task_out=leave_out_tasks,init_params=init_params) 

    
    return

def module_train_tasks(task_params,network_params,task_select,save_name_addon):
    """
    Trains output weights only on particular tasks using network
    pre-trained on pc dynamics.
    
    Args:
        task_params (tuple): Task-related parameter settiungs.
        network_params (tuple): Network related parameter settings.
        module_num_totrain (int): which module to train (1,2,3,...)

    Returns:
        None.

    """
    
    #Get parameters
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save = task_params
    state_size,module_size,num_modules,g,batch_size,\
        num_iters,nstep_sizes,init_step_size,\
            rand_key_network_init_params, save_dir,\
                pretrain_flag, train_flag, zero_out_intermoduleweights,train_recurrent_weights = network_params
    pc_select = '' #module number is equivalent to pc number
    
    input_size,output_size = task_dict[task_select]
                    
    #Create initialization parameters
    if pretrain_flag == 0:
        load_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{num_modules}_NoTrainControl_{save_name_addon}"
    else:
        load_name = f"Weightfreeze_Modular_Dunckertasks_{dim_reduction_flag}dynamicsPretrain_Input&TargetPCs_{num_modules}Modules_{state_size}Units_M{num_modules}_{save_name_addon}"
    save_name = load_name + f"_TaskTrain_{task_select}"
    file_loc_trained = save_dir+"/RNN_trained_params_"+load_name+".pickle"
    file_loc = open(file_loc_trained,'rb')
    init_params = pickle.load(file_loc)
    train_params = pickle.load(file_loc)
    fitted_pca_models = train_params[10]#get fitted pca models
    
    # file_loc_initial = save_dir+"/RNN_init_params_"+load_name+".pickle"
    # file_loc_initial = open(file_loc_initial,'rb')
    # initial_params = pickle.load(file_loc_initial)
    
    #Account for additional inputs/outputs
    init_params_new = create_rnn_params(input_size=input_size,output_size=output_size,
                                        state_size=state_size, g=g, random_key=rand_key_network_init_params)
    # init_params = init_params_new
    
    #Account for the additional input here
    init_params['change'] = np.concatenate((init_params_new['change'][:input_size,:],#init_params_new['change'][:input_size,:]
                                            init_params['change'][1:,:]),#init_params['change'][1:,:]
                                            axis = 0)
    init_params['predict'] = init_params_new['predict']#This needs to be newly drawn weights
    init_params['hidden unit'] = init_params_new['hidden unit']#init_params_new['hidden unit']


    
    #SELECT WEIGHTS TO BE TRAINED 
    #Initialize recurrent weight matrix (J)
    if train_recurrent_weights:
        weightfreeze_mask_recur = onp.concatenate((onp.ones((input_size, state_size)),
                                                onp.ones((state_size+1, state_size))),
                                                axis=0)
    else:
        weightfreeze_mask_recur = onp.concatenate((onp.zeros((input_size, state_size)),
                                                onp.zeros((state_size+1, state_size))),
                                                axis=0)
    weightfreeze_mask_dict = {}
    weightfreeze_mask_dict['hidden unit'] = np.ones((np.shape(init_params['hidden unit'])[0],
                                                     np.shape(init_params['hidden unit'])[1]))  
    
    #Input Weights PRE-Train ON
    weightfreeze_mask_recur[:input_size,:] = 1;
    
    #Output Weights Train ON   
    weightfreeze_mask_dict['change'] = np.array(weightfreeze_mask_recur)
    
    #Zero-out inter-module weights
    if zero_out_intermoduleweights:
        init_params['change'] =  ops.index_update(init_params['change'] ,ops.index[:,:],
                                         np.multiply(init_params['change'],weightfreeze_mask_dict['change']))
    
    #Predict Weights Train ON
    weightfreeze_mask_dict['predict'] = np.ones((np.shape(init_params['predict'])[0],
                                                      np.shape(init_params['predict'])[1]))
    
    train_params = (state_size,module_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,
                    init_step_size, weightfreeze_mask_dict,fitted_pca_models, 
                    optimizer_b1, optimizer_b2,optimizer_eps,pc_select,task_select,
                    rand_key_idx_task, save_dir, save_name, dropout_on,decrease_lr_on,
                    train_flag)
    print(f"Train flag:{train_flag}")
    
    #Save initial parameter settings using pickle (faster)
    filehandler = open(save_dir+"/RNN_init_params_"+save_name+'.pickle',"wb")
    pickle.dump(init_params,filehandler) 
    pickle.dump(task_params,filehandler)
    filehandler.close()
    
    #Train
    print(f'TRAIN. Task {task_select}')
    trained_params = train_weightfreeze_modular_aggregate_dynamics(train_params=train_params,task_params=task_params,leave_task_out=None,init_params=init_params) 
    
    return trained_params

def train_leave_tasks_out(task_params,network_params_pretrain,network_params_train,number_tasks_leftout):
    """
    Runs Pre and actual task training with leaving out a certain number of tasks
    in deriving the CCA dims used in Pre-training.
    
    E.g. in leave-one-out, 1 task (eg delay_pro) is left out of the bag of tasks
    used to derive CCA dims, so CCA vars used in pre-training do not take that
    left-out task into account. Then regular training is run on that left-out
    task, using the pre-trained network
    
    In the no-train control, no PRE-training is performed.
    
    Calls on module_pretrain_pcdynamics & module_train_tasks

    Args:
        task_params (tuple): task-related params.
        network_params_pretrain (tuple): pre-training network params.
        network_params_train (tuple): regular training network params.
        number_tasks_leftout (int): how many tasks are to be left out.

    Returns:
        None.

    """
    
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                    dim_reduction_flag, do_plot, do_save = task_params
               
    save_name_addon = f"Leave{number_tasks_leftout}OutPreTraining1"
    tasks_available = [key for key in task_dict]
    number_tasks = len(tasks_available)
    tasks_available = tasks_available + tasks_available
    
    task_leaveout_start_counter = 0
    task_leaveout_end_counter = task_leaveout_start_counter + number_tasks_leftout
    for task_select_counter in range(number_tasks):
        leave_out_tasks_current = []
        leave_out_tasks_current = tasks_available[task_leaveout_start_counter:task_leaveout_end_counter]
        
        #Pretrain
        for module_num_totrain in range(1,n_components_dimreduction+1):
            print(f" Pre-Train | Leave {number_tasks_leftout} out | Module Nr. {module_num_totrain}")
            module_pretrain_pcdynamics(task_params,network_params_pretrain,module_num_totrain,
                                        save_name_addon,leave_out_tasks_current)
        
        print(f"TRAIN | Leave {number_tasks_leftout} out | Current task trained:{tasks_available[task_select_counter+4]}")
        _ = module_train_tasks(task_params,network_params_train,tasks_available[task_leaveout_start_counter],save_name_addon)
        
        task_leaveout_start_counter += 1
        task_leaveout_end_counter += 1
    
    return

def run_training_dunckermethod(task_params,network_params,save_name_addon):

    
    #Get parameters
    rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                    dim_reduction_flag, do_plot, do_save = task_params
    output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
        num_iters,nstep_sizes,init_step_size,rand_key_network_init_params,\
            save_dir, train_flag,zero_out_intermoduleweights = network_params
    l2_reg_val = 1e-5
    a_reg_val = 1e-7      
    save_name_general = 'DNImethod_'
    
    covmat_update_task_counter = 1
    weightfreeze_mask_dict = {}   
    save_name_prev = ''
    rand_gen_start = rand_key_idx_task
    for key in task_dict:
        save_name = save_name_general + f"{key}" + save_name_addon
        task_type = key
        
        if covmat_update_task_counter == 1:
            init_params = create_rnn_params(input_size=input_size, output_size=output_size,
                                    state_size=state_size, g=g, random_key=rand_key_network_init_params)
            weightfreeze_mask_dict = {}
            cov_mat_prev = None
            projection_vars = None
        else:
            file_loc_trained = save_dir+"/RNN_trained_params_"+save_name_prev+".pickle"
            # file_loc_trained = save_dir+"/RNN_trained_params_"+save_name_general + "delay_pro" + save_name_addon+".pickle"
            file_loc = open(file_loc_trained,'rb')
            init_params = pickle.load(file_loc)
            cov_mat_prev = pickle.load(file_loc);print(cov_mat_prev)
            projection_vars = pickle.load(file_loc);print(projection_vars)
            rand_gen_start += num_iters*nstep_sizes
        
        save_name_prev = save_name
        covmat_update_task_counter += 1
            
       
        #####################################################################
        
        train_params = state_size,module_size,input_size,output_size,g,batch_size,num_iters,nstep_sizes,\
                init_step_size, weightfreeze_mask_dict,l2_reg_val,a_reg_val,task_type, rand_gen_start,\
                l2_reg_val,a_reg_val,save_dir, save_name, dropout_on, decrease_lr_on, train_flag,\
                    covmat_update_task_counter
        task_params = rand_gen_start, task_dict, batch_size,bval, sval, T, ntime,n_components_pca,\
                                    dim_reduction_flag, do_plot, do_save
    
        #Save using pickle (faster)
        filehandler = open(save_dir+"/RNN_init_params_"+save_name+'.pickle',"wb")
        pickle.dump(init_params,filehandler) 
        pickle.dump(train_params,filehandler) 
        pickle.dump(task_params,filehandler)
        pickle.dump(network_params,filehandler)
        filehandler.close()
        
        print(f"Run  DNI Training| Task: {task_type}")
        
        #Train RNN
        train_DNI_method(train_params=train_params,
                                        task_params=task_params,
                                        init_params=init_params,
                                        cov_mat_prev = cov_mat_prev,
                                        projection_vars_prev=projection_vars,
                                        compute_loss_alltasks_flag=False)  
    return
        
##### Run cancor task info: check how many canoncial variables are needed across tasks
# T = 1.0
# ntime = 100
# dt = T/float(ntime)
# bval = 0.1
# sval = 0.002
# output_size = 1
# input_size = 1
# n_components_dimreduction = 3
# dim_reduction_flag = 'cancor'#'cancor' #or 'pca'
# rand_key_idx_task = 1000
# do_plot = False
# do_save = False
# batch_size = 100
# save_name = 'Cancor_task_info_leavetaskout_3pcs'
# task_params = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
#                 dim_reduction_flag, do_plot, do_save
# save_info = save_dir, save_name
# r2_results_cancor = cancor_task_info(task_params,save_info)

# ########### Pre-train leaving one task out, and then train on left out task
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
output_size = 1
input_size = 1
n_components_dimreduction = 2
dim_reduction_flag = 'PCA'#'cancor' #or 'pca'
rand_key_idx_task = 1000
do_plot = False
do_save = False
batch_size = 100
task_params = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
                
#Pre-training params
state_size = 100#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.5
batch_size = 200
num_iters = 200
nstep_sizes = 5
init_step_size = 5e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
train_flag = 1
zero_out_intermoduleweights = False
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
network_params_pretrain = output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        train_flag,zero_out_intermoduleweights
                
#Training Parameters
state_size = 100#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.5
batch_size = 200#200
num_iters = 200#400
nstep_sizes = 10
init_step_size = 1e-2
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
pretrain_flag = 1
train_flag = 1
zero_out_intermoduleweights = False
train_recurrent_weights = False
network_params_train = state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        pretrain_flag,train_flag,zero_out_intermoduleweights,train_recurrent_weights


#Cycle through tasks and leave 1 out
save_name_addon = 'LeaveOneOutPreTraining_PCA1'
for task_select in task_dict:
    
    #Pretrain
    for module_num_totrain in range(1,n_components_dimreduction+1):
        print(f" Pre-Train | Leave-out {task_select} | Module Nr. {module_num_totrain}")
        module_pretrain_pcdynamics(task_params,network_params_pretrain,module_num_totrain,
                                    save_name_addon,leave_out_tasks=task_select)
    
    print(f"TRAIN | Leave-out {task_select}")
    module_train_tasks(task_params,network_params_train,task_select,save_name_addon)


#Cycle through tasks and leave 2 out
save_name_addon = 'LeaveTwoOutPreTraining_PCA1'
tasks_available = [task for task in task_dict]
tasks_available = tasks_available + tasks_available
for task_select_counter in range(len(task_dict)):
    
    leave_out_tasks_current = []
    leave_out_tasks_current = leave_out_tasks_current + [tasks_available[task_select_counter],
                                                          tasks_available[task_select_counter+1]]
    
    #Pretrain
    for module_num_totrain in range(1,n_components_dimreduction+1):
        print(f" Pre-Train | Leave-out {tasks_available[task_select_counter]},{tasks_available[task_select_counter+1]} | Module Nr. {module_num_totrain}")
        module_pretrain_pcdynamics(task_params,network_params_pretrain,module_num_totrain,
                                    save_name_addon,leave_out_tasks_current)
    
    print(f"TRAIN | Leave-out {tasks_available[task_select_counter]},{tasks_available[task_select_counter+1]}")
    module_train_tasks(task_params,network_params_train,tasks_available[task_select_counter],save_name_addon)
    


#########################################################3## PRE-TRAIN
### TASK/NETWORK PARAMS for PRE-TRAINING
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
output_size = 1
input_size = 1
n_components_dimreduction = 3
dim_reduction_flag = 'CANCOR'#'cancor' #or 'pca'
rand_key_idx_task = 1000
do_plot = False
do_save = False

#Training Parameters
state_size = 150#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.5
batch_size = 200
num_iters = 200
nstep_sizes = 5
init_step_size = 5e-3#3e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
zero_out_intermoduleweights = False
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
train_flag = 1

save_name_addon = 'Tune1'

task_params_pre = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
network_params_pre = output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        train_flag, zero_out_intermoduleweights

#Module 1 
module_pretrain_pcdynamics(task_params_pre,network_params_pre,module_num_totrain=1,save_name_addon=save_name_addon)

#Module 2
module_pretrain_pcdynamics(task_params_pre,network_params_pre,module_num_totrain=2,save_name_addon=save_name_addon)

# #Module 3
module_pretrain_pcdynamics(task_params_pre,network_params_pre,module_num_totrain=3,save_name_addon=save_name_addon)

# # #Module 4
# # # module_pretrain_pcdynamics(task_params,network_params,module_num_totrain=4,save_name_addon=save_name_addon)

########################################################### TASK TRAIN
### TASK/NETWORK PARAMS for FINAL TRAINING
# Learning rate = 1e-3 with decay works well to train recurrent weights
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
n_components_dimreduction = 3
dim_reduction_flag = 'CANCOR'#'CANCOR'
rand_key_idx_task = 1000
do_plot = False
do_save = False


#Training Parameters
state_size = 150#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.5
batch_size = 500
num_iters = 500
nstep_sizes = 10
init_step_size = 1e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
zero_out_intermoduleweights = False
train_recurrent_weights = False
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
pretrain_flag = 1
train_flag = 1

task_params = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
network_params = state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        pretrain_flag,train_flag, zero_out_intermoduleweights, train_recurrent_weights
        
save_name_addon = 'Tune1'

# TRAIN TASKS    
module_train_tasks(task_params,network_params,task_select = 'delay_pro',save_name_addon=save_name_addon)    

module_train_tasks(task_params,network_params,task_select = 'delay_anti',save_name_addon=save_name_addon)  

module_train_tasks(task_params,network_params,task_select = 'mem_pro',save_name_addon=save_name_addon)    
# # 
module_train_tasks(task_params,network_params,task_select = 'mem_anti',save_name_addon=save_name_addon)   

module_train_tasks(task_params,network_params,task_select = 'mem_dm1',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'mem_dm2',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'context_mem_dm1',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'context_mem_dm2',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'multi_mem',save_name_addon=save_name_addon) 
# 

################# TRAIN DNI (DUNCKER ET AL method)###########################
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
n_components_dimreduction = 2
dim_reduction_flag = 'CANCOR'#'CANCOR'
rand_key_idx_task = 1000
do_plot = False
do_save = False


#Training Parameters
state_size = 200#100
input_size = output_size = 3
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.3
batch_size = 400#400
num_iters = 400#400
nstep_sizes = 10
init_step_size = 1e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
zero_out_intermoduleweights = False
train_recurrent_weights = True
pretrain_flag = 0
train_flag = 1

task_params = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
network_params= output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        train_flag, zero_out_intermoduleweights

save_name_addon = 'Tune4'
run_training_dunckermethod(task_params,network_params,save_name_addon)

################ SEPARATE TRAINING #####################################
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
output_size = 1
input_size = 1
n_components_dimreduction = 2
dim_reduction_flag = 'CANCOR'#'cancor' #or 'pca'
rand_key_idx_task = 1000
do_plot = False
do_save = False

#Training Parameters
state_size = 200#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.5
batch_size = 200
num_iters = 200
nstep_sizes = 5
init_step_size = 5e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
zero_out_intermoduleweights = False
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
train_flag = 0

save_name_addon = 'Separate03'

task_params_pre = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
network_params_pre = output_size,input_size,state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        train_flag, zero_out_intermoduleweights

#Module 1 
module_pretrain_pcdynamics(task_params_pre,network_params_pre,module_num_totrain=1,save_name_addon=save_name_addon)

#Module 2
module_pretrain_pcdynamics(task_params_pre,network_params_pre,module_num_totrain=2,save_name_addon=save_name_addon)

# MAIN TRAINING
T = 1.0
ntime = 100
dt = T/float(ntime)
bval = 0.1
sval = 0.002
n_components_dimreduction = 2
dim_reduction_flag = 'CANCOR'#'CANCOR'
rand_key_idx_task = 1000
do_plot = False
do_save = False


#Training Parameters
state_size = 200#100
num_modules = n_components_dimreduction
module_size = state_size / num_modules
g = 1.3
batch_size = 500
num_iters = 500
nstep_sizes = 10
init_step_size = 3e-3
dropout_on = False
decrease_lr_on = True
optimizer_b1 = 0.9
optimizer_b2 = 0.999
optimizer_eps = 1e-8
rand_key_idx_network_init_params = 1
zero_out_intermoduleweights = False
train_recurrent_weights = True
rand_key_network_init_params = random.PRNGKey(rand_key_idx_network_init_params)
pretrain_flag = 0
train_flag = 1

task_params = rand_key_idx_task, task_dict, batch_size,bval, sval, T, ntime,n_components_dimreduction,\
                dim_reduction_flag, do_plot, do_save
network_params = state_size,module_size,num_modules,g,batch_size,\
    num_iters,nstep_sizes,init_step_size,rand_key_network_init_params, save_dir,\
        pretrain_flag,train_flag, zero_out_intermoduleweights, train_recurrent_weights
        
save_name_addon = 'Separate03'

# TRAIN TASKS    
module_train_tasks(task_params,network_params,task_select = 'delay_pro',save_name_addon=save_name_addon)    

module_train_tasks(task_params,network_params,task_select = 'delay_anti',save_name_addon=save_name_addon)  

module_train_tasks(task_params,network_params,task_select = 'mem_pro',save_name_addon=save_name_addon)    
# # 
module_train_tasks(task_params,network_params,task_select = 'mem_anti',save_name_addon=save_name_addon)   

module_train_tasks(task_params,network_params,task_select = 'mem_dm1',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'mem_dm2',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'context_mem_dm1',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'context_mem_dm2',save_name_addon=save_name_addon) 

module_train_tasks(task_params,network_params,task_select = 'multi_mem',save_name_addon=save_name_addon) 