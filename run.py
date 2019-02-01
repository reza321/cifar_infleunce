from my_cifar import CIFARModel, load_cifar
import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import experiments

num_epochs=50
batch_size=125
input_side=32
input_channels=3
weight_decay=0.001
num_classes=10
initial_learning_rate=0.01
damping=1e-2
keep_probs = [1.0, 1.0]
decay_epochs = [10000, 20000]
kernel_size=3
filter1=8
filter2=16
dense1_unit=32
dense2_unit=8

data=load_cifar()
#datam=data.test.x.reshape(data.test.x.shape[0],32,32,3).astype(np.uint8)

model=CIFARModel(
    weight_decay = weight_decay,
	initial_learning_rate=initial_learning_rate,
	num_classes=num_classes, 
	input_side=input_side,
	input_channels=input_channels,
	data_sets=data,
	num_epochs=num_epochs,
	batch_size=batch_size,
	decay_epochs=decay_epochs,	
	kernel_size=kernel_size,	
	dense1_unit=dense1_unit,
	dense2_unit=dense2_unit,
	filter1=filter1,
	filter2=filter2,
	train_dir='output', 
	log_dir='log',
	model_name='cifar_mymodel',
	damping=1e-2)

num_steps = 128000

run_phase='all'
if run_phase=='all':
	model.train(num_steps=num_steps,iter_to_switch_to_batch=200000,
    iter_to_switch_to_sgd=400000)


iter_to_load = num_steps -1

test_idx = 6558

if run_phase=='all':
	known_indices_to_remove=[]
else:
	f=np.load('output/cifar_mymodel_300K_retraining_3pts.npz')
	known_indices_to_remove=f['indices_to_remove']


actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
	model, 
	test_idx=test_idx, 
	iter_to_load=iter_to_load, 
	num_to_remove=3,
	num_steps=10000, 
	remove_type='maxinf',
	known_indices_to_remove=known_indices_to_remove,
	force_refresh=True)

filename="cifar_mymodel"+str(num_steps)+"_"+run_phase+".txt"
np.savetxt(filename, np.c_[actual_loss_diffs,predicted_loss_diffs],fmt ='%f6')

if run_phase=="all":
	np.savez(
		'output/cifar_mymodel_300K_retraining-3pts.npz', 
		actual_loss_diffs=actual_loss_diffs, 
		predicted_loss_diffs=predicted_loss_diffs, 
		indices_to_remove=indices_to_remove
	)








































