import tensorflow as tf
import numpy as np
from sklearn import metrics
import sys


"""
Variant of binary cross-entropy loss which is zero for missing labels.
"""
def multi_task_binary_classification_loss(y_true, y_pred):
  label_flag = y_true[:,1]
  labels = y_true[:,0]
  return tf.keras.backend.binary_crossentropy(labels, y_pred[:,0]) * label_flag

  
"""
Variant of categorical cross-entropy loss which is zero for missing labels.
"""
def multi_task_categorical_classification_loss(y_true, y_pred):
  label_flag = y_true[:,1]
  labels = y_true[:,0]
  #labels = tf.Print(labels, [labels], message='M LABELS=', summarize=100)
  #label_flag = tf.Print(label_flag, [label_flag], message='M FLAGS=', summarize=100)
  #y_pred = tf.Print(y_pred, [y_pred], message='M PRED', summarize=100)
  #labels_tmp = tf.where(tf.less(labels, 0), 0, labels)
  return tf.keras.backend.sparse_categorical_crossentropy(labels, y_pred) * label_flag


"""
Variant of mean squared error loss which is zero for missing labels.
"""
def multi_task_regression_loss(y_true, y_pred):
  label_flag = y_true[:,1]
  labels = y_true[:,0]
  return tf.keras.losses.mean_squared_error(labels, y_pred[:,0]) * label_flag
  
    
class MultiTargetModel:


  """
  Initialize the multi-target model for the given task and training options.
  """
  def __init__(self, num_features, tasks, opts):
    self.iteration = 0
    self.tasks = tasks
    self.num_features = num_features
    self.create_model(opts)
    
   
  def create_model(self, opts):
    self.iteration = 0
  
    hidden_layers = []
    loss_list = []
    
    hidden_sizes = opts['hidden_sizes']
    num_shared = opts['num_shared_hidden_layers'] if 'num_shared_hidden_layers' in opts else len(hidden_sizes)
    l2_reg = opts['l2_regularization'] if 'l2_regularization' in opts else 0.0
    dropout_rate = opts['dropout_rate'] if 'dropout_rate' in opts else 0.0
    
    def get_output_layer(task, num_output, activation, l2_reg):
      return tf.keras.layers.Dense(num_output, 
          activation=activation,
          kernel_initializer=tf.keras.initializers.VarianceScaling(), 
          kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None, 
          name="output_for_" + task)      
    
    # create multi-task model
    input_layer = tf.keras.layers.Input(shape=(self.num_features,))
    previous_layer = input_layer
    for layer_index, layer_size in enumerate(hidden_sizes[:num_shared]):
      hidden_layer = tf.keras.layers.Dense(layer_size, 
        activation='relu', 
        kernel_initializer=tf.keras.initializers.VarianceScaling(),  
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None, 
        name="hidden_layer_" + str(layer_index))(previous_layer)
      if dropout_rate > 0:
        hidden_layer = tf.keras.layers.Dropout(dropout_rate)(hidden_layer, training=True)
      hidden_layers.append(hidden_layer)
      previous_layer = hidden_layers[-1]
    output_layers = []
    for (task, num_classes) in self.tasks:
      previous_layer = hidden_layers[-1]
      # Add task-specific hidden layers if not all hidden layers are shared
      for layer_index, layer_size in enumerate(hidden_sizes[num_shared:]):
        task_specific_layer = tf.keras.layers.Dense(layer_size, 
          activation='relu', 
          kernel_initializer=tf.keras.initializers.VarianceScaling(),  
          kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None, 
          name="task_specific_layer_" + task + "_" + str(layer_index))(previous_layer)
        if dropout_rate > 0:
          task_specific_layer = tf.keras.layers.Dropout(dropout_rate)(task_specific_layer, training=True)
        previous_layer = task_specific_layer
      if num_classes == 0:
        # linear output layer for regression
        output_layers.append( get_output_layer(task, 1, None, l2_reg) (previous_layer) )
        loss_list.append(multi_task_regression_loss)
      elif num_classes == 2:
        # sigmoid output layer for (binary) classification
        output_layers.append( get_output_layer(task, 1, tf.nn.sigmoid, l2_reg) (previous_layer) )
        loss_list.append(multi_task_binary_classification_loss)
      elif num_classes > 2:
        # softmax output layer for multi-class classification
        output_layers.append( get_output_layer(task, num_classes, tf.nn.softmax, l2_reg) (previous_layer) )
        loss_list.append(multi_task_categorical_classification_loss)
      else:
        raise ValueError("Cannot have num_classes = %d in classification task for training DNN" % num_classes)
                  
    self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layers)
    
    learning_rate = opts['learning_rate'] if 'learning_rate' in opts else 0.01
    decay = opts['learning_rate_decay'] if 'learning_rate_decay' in opts else 0.0
    momentum = opts['momentum'] if 'momentum' in opts else 0.9
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
    self.model.compile(optimizer=sgd, loss=loss_list, metrics=None)
    
  
  """
  Train the model on the given dataset.
  """
  def train(self, dataset, opts):
    # exclude the completely unlabeled instances
    labels_and_flags = dataset.get_labels_and_flags()
    all_flags = np.hstack([np.expand_dims(f[:,1], axis=-1) for f in labels_and_flags])
    labeled_indices = np.sum(all_flags, axis=1) > 0
    #print("--- MEAN of training labels:", [np.mean(lf[lf[:,1]!=0,0]) for lf in labels_and_flags])
    #print("--- STD of training labels:", [np.std(lf[lf[:,1]!=0,0]) for lf in labels_and_flags])
    self.model.fit(
      dataset.features[labeled_indices,:],
      [lf[labeled_indices,:] for lf in labels_and_flags], 
      epochs=opts['num_epochs'] if self.iteration == 0 else opts['cl_retrain_num_epochs'], 
      batch_size=opts['batch_size']
    )
    self.iteration += 1
    
    
  """
  Perform predictions for each task in a dataset.
  """
  def predict(self, dataset, scores=True):
    predictions = self.model.predict(dataset.features)
    if not scores:
      for task_index in range(len(self.tasks)):
        predictions[task_index] = self.scores_to_labels(predictions[task_index], 
          num_classes=self.tasks[task_index][1])
    return predictions
  

  """
  Perform predictions for each task in a dataset, using multiple passes w/ dropout.
  """
  def predict_with_dropout(self, dataset, opts, scores=True, verbose=True, random=False):
    def safe_log(x):
      return np.log(np.maximum(x, 1e-10))
      
    model = self.model
    x_test = dataset.features
    num_inst = x_test.shape[0]
    all_pred = []
    n_passes = opts['num_dropout_passes']
    assert n_passes >= 1, "number of dropout passes must be >= 1"
    is_classification = [t[1] > 0 for t in self.tasks]
    for p in range(n_passes):
      pred = model.predict(x_test)
      if not isinstance(pred, list):  # single-task outputs
        pred = [pred]
      num_tasks = len(pred)
      assert num_tasks == len(self.tasks), "unexpected # of tasks in prediction"
      for task_index in range(num_tasks):
        num_outp = pred[task_index].shape[1]
        if p == 0:
          all_pred.append(np.zeros((n_passes, num_inst, num_outp)))
        all_pred[task_index][p, ...] = pred[task_index]
      if verbose:
        sys.stdout.write(".")
        sys.stdout.flush()
    if verbose:
      sys.stdout.write("\n")
    pred = [None] * num_tasks
    uncertainty = [None] * num_tasks
    for task_index in range(num_tasks):
      pred[task_index] = np.mean(all_pred[task_index], axis=0)
      num_outp = pred[task_index].shape[1]
      if random:
        print("--- RANDOMIZING the uncertainties")
        uncertainty[task_index] = np.random.uniform(size=pred[task_index].shape[0])
      else:
        if is_classification[task_index]:
          # classification: predictive entropy
          if num_outp == 1: # binary
            uncertainty[task_index] = -(pred[task_index] * safe_log(pred[task_index]) + (1-pred[task_index]) * safe_log(1-pred[task_index]))
            uncertainty[task_index] = np.squeeze(uncertainty[task_index], axis=1)
          else: # multi-class (softmax)
            uncertainty[task_index] = -np.sum( pred[task_index] * safe_log(pred[task_index]), axis=1 )
        else:
          # regression: standard deviation
          uncertainty[task_index] = np.squeeze(np.std(all_pred[task_index], axis=0), axis=1)
      if not scores:
        pred[task_index] = self.scores_to_labels(pred[task_index], num_classes=self.tasks[task_index][1])
    return pred, uncertainty
    
    
  """
  Converts scores to discrete labels for classification tasks.
  """
  def scores_to_labels(self, predictions, num_classes):
    if num_classes == 0:
      labels = predictions[:,0] # TODO: apply inverse scaling
    elif num_classes == 2:
      labels = (predictions > 0.5).astype(np.int64)
      labels = np.squeeze(labels, axis=1)
      #[1 if score > 0.5 else 0 for score in predictions]
    elif num_classes > 2:
      labels = np.argmax(predictions, axis=1)
    else:
      raise ValueError("num_classes = %d not allowed" % num_classes)
    return labels


  """
  Predict on an evaluation set and compute the performance of the model.
  """
  def predict_and_evaluate(self, dataset, opts):
    labels_and_flags = dataset.get_labels_and_flags()
    if opts['use_dropout_for_eval']:
      all_predictions, _ = self.predict_with_dropout(dataset, opts)
    else:
      all_predictions = self.predict(dataset)
    for task_index, (task, num_classes) in enumerate(self.tasks):
      labels_defined = labels_and_flags[task_index][:,1] != 0
      predictions = self.scores_to_labels(all_predictions[task_index][labels_defined,:], num_classes)
      labels = labels_and_flags[task_index][labels_defined,0]
      #print("--- TEST MEAN:", np.mean(predictions))
      #print("--- TEST STD:", np.std(predictions))
      if num_classes == 0: # regression
        mae = np.mean(np.abs(predictions - labels))
        print("MAE for task", task, ":", mae)
        cc = np.corrcoef(predictions, labels)[0,1]
        print("CC for task", task, ":", cc)
      else:
        uar = metrics.recall_score(labels, predictions, pos_label = None, average='macro')
        print("UAR for task", task, ":", uar)

