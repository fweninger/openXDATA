import numpy as np
import arff
import sklearn.preprocessing


class MultiTargetDataset:


  """
  Initialize the dataset.
  """
  def __init__(self):
    self.instance_names = None
    self.features = None
    self.instances_per_file = []
    self.labels = []
    self.feature_names = []
    self.feature_types = []
    self.target_names = []
    self.target_types = []
    self.num_classes = []
    self.target2index = {}
    self.datafile_scales = []
    self.global_scale = None
    self.label_scales = []
  
    
  """
  Initializes the dataset from a spec file in CSV format, 
  which contains the names of ARFF files and the # of target attributes per file.
  """
  def init_from_spec(self, spec_file):
    fh = open(spec_file)
    for line in fh:
      if line.startswith("#") or line == "":
        continue
      els = line.split(",")
      if len(els) != 2:
        raise ValueError("Invalid line in spec file: must contain two comma-separated values")
      arff_file, num_targets = els[0], int(els[1])
      self.add_arff(arff_file, num_targets)
    fh.close()

    
  """
  Add data from ARFF file. Supports multi-label data, missing labels, 
  and completely unlabeled data (no target attributes).
  """
  def add_arff(self, arff_file, num_targets=1):
    fh = open(arff_file)
    arff_data = arff.load(fh, encode_nominal=True)
    fh.close()
    arff_values = np.array(arff_data['data'])

    # determine attributes and targets
    feature_indices = []
    target_indices = []
    feature_names = []
    target_names = []
    feature_types = []
    target_types = []
    has_instance_name = False
    for attr_index, (attr_name, attr_type) in enumerate(arff_data['attributes']):
      if attr_type == 'STRING':
        # assume first string attribute is instance name
        if attr_index == 0:
          has_instance_name = True
        # ignore all string attributes
        continue
      if attr_index >= len(arff_data['attributes']) - num_targets:
        if attr_name in target_names:
          raise ValueError("Target %s is duplicated in arff file %s" % (attr_name, arff_file))
        target_indices.append(attr_index)
        target_names.append(attr_name)
        target_types.append(attr_type)
      else:
        feature_indices.append(attr_index)
        feature_names.append(attr_name)
        feature_types.append(attr_type)

    # check feature names and types
    if self.feature_names:
      if feature_names != self.feature_names:
        raise ValueError("Feature names in arff file %s are not compatible with other features: %s != %s" % (arff_file, feature_names, self.feature_names))
    else:
      self.feature_names = feature_names
    if self.feature_types:
      if feature_types != self.feature_types:
        raise ValueError("Feature types in arff file %s are not compatible with other features: %s != %s" % (arff_file, feature_types, self.feature_types))
    else:
      self.feature_types = feature_types

    # append instance names
    if has_instance_name:
      instance_names = arff_values[:,0].astype(np.str)
    else:
      instance_names = np.array([arff_data['relation'] + '_instance_' + str(i+1) \
                                for i in range(arff_values.shape[0])])
    if self.instance_names is None:
      self.instance_names = instance_names
    else:
      self.instance_names = np.concatenate((self.instance_names, instance_names))

    # get features
    new_data = arff_values[:,feature_indices].astype(np.float64)

    # append features
    if self.features is None:
      prev_num_instances = 0
      self.features = new_data
    else:
      prev_num_instances = self.features.shape[0]
      self.features = np.vstack((self.features, new_data))
    num_instances = self.features.shape[0]
    self.instances_per_file.append(range(prev_num_instances, num_instances))

    # generate multi-task labels (one column per task)
    for target_index, target_name, target_type in zip(target_indices, target_names, target_types):
      new_labels = arff_values[:,target_index]
      # existing task --> append new labels to existing label column
      if target_name in self.target_names:
        idx = self.target2index[target_name]
        if self.target_types[idx] != target_type:
          raise ValueError("Target type mismatch in arff file %s for target %s: %s vs. %s" \
            % (arff_file, target_name, target_type, self.target_types[idx]))
        self.labels[idx] = np.concatenate((self.labels[idx], new_labels))
      # new task --> create new label column
      else:
        self.target_names.append(target_name)
        if prev_num_instances > 0:
          self.labels.append(np.concatenate(([None] * prev_num_instances, new_labels)))
        else:
          self.labels.append(new_labels)
        self.target2index[target_name] = len(self.labels) - 1
        self.target_types.append(target_type)
        self.num_classes.append(len(target_type) if isinstance(target_type, list) else 0)
    # pad label columns to the same length
    for target_index in range(len(self.labels)):
      num_labels = self.labels[target_index].shape[0]
      if num_labels != num_instances:
        self.labels[target_index] = np.concatenate((self.labels[target_index], [None] * (num_instances - num_labels)))


  """
  Write multi-target dataset to ARFF file. Supports missing labels.
  """
  def write_arff(self, arff_file, relation='multi_target'):
    # get string representation of numeric labels (class index or regression target)
    def encode_labels(labels, label_type):
      if str(label_type).upper() == 'NUMERIC':
        return labels
      elif isinstance(label_type, list): # nominal
        labels_numeric = np.copy(labels) # copy
        labels_numeric[labels == None] = -1
        label_type_with_missing = label_type + ['?']
        # note: can't index by object type array
        return np.take(label_type_with_missing, labels_numeric.astype(np.int64))
      else:
        raise ValueError("label_type = '%s' not allowed" % label_type)
        
    fh = open(arff_file, "w")
    arff_data = {
      'data': np.hstack((np.expand_dims(self.instance_names, axis=-1),
                         self.features,
                         np.hstack([np.expand_dims(encode_labels(self.labels[idx], self.target_types[idx]), axis=-1) \
                           for idx in range(len(self.labels))])
                       )),
      'attributes': [('instance_name', 'STRING')] \
        + list(zip(self.feature_names, self.feature_types)) \
        + list(zip(self.target_names, self.target_types)),
      'relation': relation, 
      'description': 'multi-target dataset generated by openXData 0.1'
    }
    arff.dump(arff_data, fh)
    fh.close()
    
  
  """
  Determine if the task with the given index is a classification task.
  """
  def is_classification(self, task_index):
    return str(self.target_types[task_index]).upper() != 'NUMERIC'
    

  """
  Process the features of the dataset, applying standardization per data file.
  """    
  def standardize_features_per_datafile(self):
    num_datafiles = len(self.instances_per_file)
    self.datafile_scales = []
    for i in range(num_datafiles):
      scaler = sklearn.preprocessing.StandardScaler()
      self.features[self.instances_per_file[i],:] = \
        scaler.fit_transform(self.features[self.instances_per_file[i],:])
      self.datafile_scales.append(scaler)

    
  """
  Process the features of a test dataset, applying standardization from the training set per data file.
  """
  def standardize_features_per_datafile_test(self, test_dataset):
    num_datafiles = len(self.instances_per_file)
    num_datafiles_other = len(test_dataset.instances_per_file)
    if num_datafiles != num_datafiles_other:
      raise ValueError("Number of data files does not match: %d vs. %d" % \
        (num_datafiles, num_datafiles_other))
    if len(self.datafile_scales) != num_datafiles_other:
      raise ValueError("Number of scales files does not match number of data files: %d vs. %d" % \
        (len(self.datafile_scales), num_datafiles_other))
    for i in range(num_datafiles):
      test_dataset.features[test_dataset.instances_per_file[i],:] = \
        self.datafile_scales[i].transform(test_dataset.features[test_dataset.instances_per_file[i],:])
    
    
  """
  Process the features of the dataset, applying global standardization.
  """
  def standardize_features_global(self):
    scaler = sklearn.preprocessing.StandardScaler()
    self.features = scaler.fit_transform(self.features)
    self.global_scale = scaler
    
    
  """
  Process the features of a test dataset, applying global standardization from the training set.
  """
  def standardize_features_global_test(self, test_dataset):
    if self.global_scale is None:
      raise ValueError("Global scale has not been computed!")
    test_dataset.features = self.global_scale.transform(test_dataset.features)


  """
  Standardize the regression labels for each task.
  """
  def standardize_labels(self):
    num_tasks = len(self.labels)
    self.label_scales = []
    for task_index in range(num_tasks):
      if self.is_classification(task_index):
        scaler = None
      else:
        defined_indices = self.labels[task_index] != None
        scaler = sklearn.preprocessing.StandardScaler()
        std_labels = scaler.fit_transform(self.labels[task_index][defined_indices].reshape(-1, 1).astype(np.float64))
        #print("DEBUG STD:", np.std(std_labels[:,0]))
        self.labels[task_index][defined_indices] = std_labels[:,0]
      self.label_scales.append(scaler)
      

  """
  Process the labels of a test dataset, applying standardization from the current dataset.
  """
  def standardize_labels_test(self, test_dataset):
    num_tasks = len(self.labels)
    if len(test_dataset.labels) != num_tasks:
      raise ValueError("Datasets must have the same # of tasks")
    for task_index in range(num_tasks):
      if not self.is_classification(task_index):
        defined_indices = test_dataset.labels[task_index] != None
        std_labels = self.label_scales[task_index].transform(test_dataset.labels[task_index][defined_indices].reshape(-1, 1).astype(np.float64))
        test_dataset.labels[task_index][defined_indices] = std_labels[:,0]

        
  """
  Invert the standardization for the regression labels of each task.
  """
  def unstandardize_labels(self):
    num_tasks = len(self.labels)
    if len(self.label_scales) != num_tasks:
      raise ValueError("Unexpected number of label scales, expected the number of tasks (%d)" % num_tasks)
    for task_index in range(num_tasks):
      if self.is_classification(task_index):
        continue
      else:
        defined_indices = self.labels[task_index] != None
        scaler = self.label_scales[task_index]
        unstd_labels = scaler.inverse_transform(self.labels[task_index][defined_indices].reshape(-1, 1).astype(np.float64))
        self.labels[task_index][defined_indices] = unstd_labels[:,0]
    
      
  """
  Return a list of pairs (task_name, num_classes) for each task in the dataset.
  """
  def get_tasks(self):
    return list(zip(self.target_names, self.num_classes))
    
    
  """
  Return the number of features in the dataset.
  """
  def get_num_features(self):
    return self.features.shape[1]
    
  
  """
  Return a list of numpy matrices, one for each task, where the first column is 
  the numeric label and the second column is the label presence flag (0 = missing label).
  """
  def get_labels_and_flags(self):
    ret = []
    num_tasks = len(self.labels)
    for task_index in range(num_tasks):
      label_type = np.int64 if self.is_classification(task_index) else np.float64
      labels = np.copy(self.labels[task_index])
      labels[labels == None] = 0  # dummy label
      flags = (self.labels[task_index] != None).astype(label_type)
      labels_flags = np.hstack((
        np.expand_dims(labels.astype(label_type), axis=-1), 
        np.expand_dims(flags, axis=-1)
      ))
      ret.append(labels_flags)
    return ret
    
    
  """
  Update the missing labels of the dataset w/ the most certain predictions for each task.
  """
  def update_labels_with_predictions(self, predictions, uncertainty, num_instances_to_label, standardize_predictions=False):
  
    num_tasks = len(self.labels)
    assert len(predictions) == num_tasks, "mismatched number of tasks in predictions"
    assert len(uncertainty) == num_tasks, "mismatched number of tasks in predictions"
    #assert percentage > 0 and percentage <= 1, "percentage must be in (0,1]"
    assert num_instances_to_label > 0
    total_unlabeled_instances = 0

    for task_index in range(num_tasks):
    
      num_instances = predictions[task_index].shape[0]
      assert predictions[task_index].shape[0] == self.labels[task_index].shape[0], "mismatch between number of predictions and labels"

      if standardize_predictions and not self.is_classification(task_index):
        scaler = sklearn.preprocessing.StandardScaler()
        ptask = scaler.fit_transform(predictions[task_index].reshape(-1, 1))[:,0]
      else:
        ptask = predictions[task_index]

      unlabeled_instances = list(filter(lambda i: self.labels[task_index][i] is None, range(num_instances)))
      # most certain instances out of the unlabeled ones
      num_selected_instances = min(num_instances_to_label, len(unlabeled_instances))
      print("Added labels for", num_selected_instances, "/", len(unlabeled_instances), "unlabeled instances for task", self.target_names[task_index])
      selected_instances = sorted(unlabeled_instances, \
        key=lambda i: uncertainty[task_index][i])[:num_selected_instances]
      predicted_labels = ptask[selected_instances]
      self.labels[task_index][selected_instances] = predicted_labels
      #print("Selected instances:", selected_instances, \
      #  "with uncertainties =", uncertainty[task_index][selected_instances])
      total_unlabeled_instances += (len(unlabeled_instances) - num_selected_instances)

    return total_unlabeled_instances
     

