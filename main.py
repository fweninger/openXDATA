#!/usr/bin/python


from dataset import MultiTargetDataset
from model import MultiTargetModel
from options import DEFAULT_OPTS

import tensorflow as tf
import numpy as np

import sys
import yaml
import random


# read configuration options (iterations, batch size, etc.) from yaml file
if len(sys.argv) != 2:
  print("Usage: %s [config_file]" % sys.argv[0])
  sys.exit(1)
  
config_file = open(sys.argv[1], 'r')
config_opts = yaml.load(config_file, Loader=yaml.Loader)
opts = dict(DEFAULT_OPTS)
opts.update(config_opts)
if opts['display_config']:
  print("--- OPTIONS:")
  for k, v in sorted(opts.items()):
    print("%s = %s" % (k, v))

if opts['cl_spec'] is None:
  print("ERROR: cl_spec must be set in config file")
  sys.exit(1)

# set random seeds (Python, numpy, tensorflow)
rs = opts['random_seed']
random.seed(rs)
np.random.seed(rs)
tf.set_random_seed(rs)

# create cross-labeling dataset from spec
cl_dataset = MultiTargetDataset()
cl_dataset.init_from_spec(opts['cl_spec'])

if opts['initial_dataset_name']:
  cl_dataset.write_arff(opts['initial_dataset_name'] + '.arff', relation=opts['initial_dataset_name'])

if opts['standardize_features_per_datafile']:
  cl_dataset.standardize_features_per_datafile()
if opts['standardize_features_global']:
  cl_dataset.standardize_features_global()

# create evaluation dataset from spec
eval_dataset = None
if opts['eval_spec'] is not None:
  eval_dataset = MultiTargetDataset()
  eval_dataset.init_from_spec(opts['eval_spec'])
  if opts['standardize_features_per_datafile']:
    if opts['standardize_test_features_on_train']:
      cl_dataset.standardize_features_per_datafile_test(eval_dataset)
    else:
      eval_dataset.standardize_features_per_datafile()
  if opts['standardize_features_global']:
    if opts['standardize_test_features_on_train']:
      cl_dataset.standardize_features_global_test(eval_dataset)
    else:
      eval_dataset.standardize_features_global()
  
# standardize the labels for regression tasks on both datasets
if opts['standardize_labels']:
  cl_dataset.standardize_labels()
  if eval_dataset is not None:
    cl_dataset.standardize_labels_test(eval_dataset)

# create a MTL model w/ the tasks from the cross-labeling dataset
tasks = cl_dataset.get_tasks()
if opts['display_tasks']:
  print("--- TASKS:")
  for t, c in tasks:
    print(t + ": " + ("classification (%d)" % c if c else "regression"))
mtl_model = MultiTargetModel(cl_dataset.get_num_features(), tasks, opts)

# train the MTL model w/ the cross-labeling dataset features and the dataset label+flag matrices
print("*** Training a model on the initial training set ...")
mtl_model.train(cl_dataset, opts)

for iteration in range(opts['num_cl_iters']):
  # evaluate on the evaluation dataset (compute performance per task excl. missing labels)
  if eval_dataset is not None:
    print("*** Evaluating the current model ...")
    mtl_model.predict_and_evaluate(eval_dataset, opts)
  # predict the cross-labeling dataset
  print("*** Performing predictions for unlabeled training data ...")
  predictions, uncertainty = mtl_model.predict_with_dropout(cl_dataset, opts, scores=False, random=opts['cl_randomize_selection'])
  # update the labels in the cross-labeling dataset
  num_unlabeled = cl_dataset.update_labels_with_predictions(predictions, uncertainty, opts['cl_inst_per_iter'], opts['standardize_predictions'])
  # retrain model
  if opts['cl_retrain_from_scratch']:
    mtl_model.create_model(opts)
  mtl_model.train(cl_dataset, opts)
  if num_unlabeled == 0:
    print("*** No unlabeled instances remain, terminating the cross-labeling process.")
    break

# final evaluation
if eval_dataset is not None:
  print("*** Evaluating the final model ...")
  mtl_model.predict_and_evaluate(eval_dataset, opts)
  
# write the cross-labeled dataset
if opts['standardize_labels']:
  cl_dataset.unstandardize_labels()
print("*** Writing the cross-labeled dataset ...")
cl_dataset.write_arff(opts['final_dataset_name'] + '.arff', relation=opts['final_dataset_name'])
