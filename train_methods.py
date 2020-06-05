import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hypergrad import SGDHD, AdamHD
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from config import SimpleNet,ThreeLayer
import bcolz
import pandas as pd
import os.path as osp
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials, rand



from config import (get_optimizer, DATA_DIR, MODEL_CLASS, LOSS_FN,
                    HYPERPARAM_NAMES, EPOCHS, BATCH_SIZE, POPULATION_SIZE,
                    EXPLOIT_INTERVAL, USE_SQLITE, SPACE, METHOD_SHORT)
                    

def train(model, optimizer, loss_fn, x, y, hypergrad, epochs=1, device='cpu'):
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	num_correct, num_samples = 0, 0
	iters = 70
	check = 10
	loss_history=[]
	acc_history = []
	lr_history = []
	for e in range(epochs):
		for i in range(iters):
			model.train()
			optimizer.zero_grad()
			# print(e, i, optimizer.param_groups[0]['lr'])
			out = torch.squeeze(model(x))
			# print(out, y)

			loss = loss_fn(out, y)
			loss.backward()
			optimizer.step()
			if i%check==0:
				# print('Epoch: %d, Iters: %d, loss = %.4f' % (e, i, loss))
				acc = check_accuracy(model, x, y, device)
			loss_history.append(loss)
			acc_history.append(acc)
			if hypergrad:
				if e==0 and i==0:
					lr_history.append(optimizer.param_groups[0]['lr'])
				else:
					lr_history.append(optimizer.param_groups[0]['lr'].item())
	return (loss_history, acc_history, lr_history)

def check_accuracy(model, x, y, device):

	model.eval()  # set model to evaluation mode
	with torch.no_grad():
		x = x.to(device=device)  # move to device, e.g. GPU
		y = y.to(device=device)
		preds = torch.squeeze(model(x) > 0.5)
		num_correct = (preds == y).sum()
		num_samples = preds.size(0)
		acc = float(num_correct) / num_samples
		# print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
	return acc

def bayesian_optim(space, plot=True):
	trials = Trials()
	tpe_algo = tpe.suggest
	tpe_algo = tpe.suggest
	best = fmin(objective_func,
			space = space, 
			algo=tpe.suggest, 
			max_evals = 50, 
			trials=trials)
	tpe_results = pd.DataFrame({'acc': [x['loss'] for x in trials.results], 
								'iteration': trials.idxs_vals[0]['learning_rate'],
								'learning_rate': trials.idxs_vals[1]['learning_rate']})
	# print(tpe_results)
	if plot:
		bayes_plot(tpe_results, best)
	return best

def objective_func(space):
	inputs = bcolz.open(osp.join(DATA_DIR, "trn_inputs.bcolz"), 'r')
	targets = bcolz.open(osp.join(DATA_DIR, "trn_targets.bcolz"), 'r')
	# print(space)
	inputs = torch.Tensor(inputs)
	targets = torch.Tensor(targets)
	net = MODEL_CLASS()
	loss = LOSS_FN
	optimizer = optim.Adam(net.parameters(), lr=space['learning_rate']) #to use hypergrad descent use SGDHD or AdamHD 

	loss_history, acc_history, lr_history = train(net, optimizer, loss, inputs, targets, False, EPOCHS)
	final_loss = loss_history[-1]
	return {'loss': -acc_history[-1], 'status':STATUS_OK}

def bayes_plot(tpe_results, best):
	plt.clf()
	plt.subplot(2, 1, 1)
	plt.plot(tpe_results['iteration'], [best['learning_rate']]*len(tpe_results['iteration']), color = 'r')
	plt.scatter(tpe_results['iteration'],tpe_results['learning_rate'])

	plt.title(METHOD + ': Sequence of Values Sampled')
	plt.xlabel('Iteration')
	plt.ylabel('Learning Rate')
	plt.subplot(2, 1, 2)
	best_acc = [np.max(np.abs(tpe_results['acc'][:i])) for i in range(len(tpe_results['acc']))]

	plt.plot(best_acc*100)
	plt.title(METHOD + ' Convergence for ' + NET_NAME + ' on Dataset ' + str(DATA_NUM))
	plt.xlabel('Trial')
	plt.ylabel('Best Model Accuracy')
	plt.tight_layout()
	filename = METHOD_SHORT + '_NN' + str(NET_NUM) + '_D' + str(DATA_NUM) 
	plt.savefig(filename)




def loss_plot(loss_history, acc_history, lr_history=None):# Plot the loss function and train / validation accuracies
	plt.subplot(3, 1, 1)
	plt.plot(loss_history)
	plt.title('HGD Loss History')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.subplot(3, 1, 2)
	plt.plot(acc_history*100, label='train')
	# plt.plot(stats['val_acc_history'], label='val')
	plt.title('HGD Classification Accuracy History')
	plt.xlabel('Iteration')
	plt.ylabel('Classification accuracy')
	# plt.legend()

	# plt.plot(stats['val_acc_history'], label='val')
	if lr_history:
		plt.subplot(3, 1, 3)
		plt.plot(lr_history, label='learning_rate')
		plt.title('HGD Learning Rate History')
		plt.xlabel('Iteration')
		plt.ylabel('Learning rate')
	plt.tight_layout()
	plt.savefig('HGD_NN2_D3')

def hypergrad(lr=0.001, momentum=1, plot=True):
	inputs = bcolz.open(osp.join(DATA_DIR, "trn_inputs.bcolz"), 'r')
	targets = bcolz.open(osp.join(DATA_DIR, "trn_targets.bcolz"), 'r')
	inputs = torch.Tensor(inputs)
	targets = torch.Tensor(targets)
	net = MODEL_CLASS()
	loss = LOSS_FN
	optimizer = AdamHD(net.parameters(), lr=lr, hypergrad_lr=0.001) #to use hypergrad descent use SGDHD or AdamHD 
	lr = optimizer.param_groups[0]['lr']
	loss_history, acc_history, lr_history = train(net, optimizer, loss, inputs, targets, True, EPOCHS)
	loss_plot(loss_history, acc_history, lr_history)
	return(loss_history[-1], acc_history[-1])

if __name__ == "__main__":
	if METHOD_SHORT == 'RS':
		METHOD = 'Random Search'
	elif METHOD_SHORT == 'Bayes':
	    METHOD = 'Bayesian Optimization'
	if METHOD_SHORT == 'HGD':
		METHOD = 'Hypergradient Descent'
	for DATA_NUM in range(1,4):
		if DATA_NUM == 1:
		    DATA_DIR = "../../data/lin"
		elif DATA_NUM == 2:
		    DATA_DIR = "../../data/simple"
		elif DATA_NUM == 3:
		    DATA_DIR = "../../data/threelayer"

		for NET_NUM in range(2,4):
			if NET_NUM == 1:
			    NET_NAME = 'One Layer NN'
			    MODEL_CLASS = Nron
			elif NET_NUM == 2:
			    NET_NAME = 'Simple NN'
			    MODEL_CLASS = SimpleNet
			elif NET_NUM == 3:
			    NET_NAME = 'Three Layer NN'
			    MODEL_CLASS = ThreeLayer

		
			print(bayesian_optim(SPACE))
			# print(hypergrad())


