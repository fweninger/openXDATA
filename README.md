# openXDATA

openXDATA is an open-source tool for multi-target data generation and missing label completion.
Given a set of feature files with labels for one or more tasks (missing labels are allowed), the tool generates a feature file where all instances are labeled in all the tasks and missing labels are completed.
The underlying algorithm is cross-data label completion (CDLC), which is based on iterative pseudo-labeling with multi-task shared-hidden-layer neural networks.
More information about the algorithm can be found in the paper referenced below.
The implementation is done using the Keras API of TensorFlow.


## Citing

If you use openXDATA for your research, please cite the following paper:

Felix Weninger, Yue Zhang, Rosalind W. Picard, "openXDATA: A Tool for Multi-Target Data Generation and Missing Label Completion", Journal of Machine Learning Research (submitted)


## Installation

In order to use openXDATA, you need Python 3.x. 
We suggest using a Python3 virtual environment to avoid version conflicts:

    virtualenv /path/to/virtualenv
    source /path/to/virtualenv/bin/activate

Inside the virtual environment, you need to install the following packages using pip:

    pip install tensorflow==1.12.0
    pip install liac-arff
    pip install sklearn
    pip install pyyaml


## Running openXDATA

openXDATA can be run by executing main.py with a configuration file in YAML format:

    python main.py config.yaml
    
In the `example` directory, there is an example configuration file along with data files. It can be run using

    cd example
    python ../main.py config.yaml


## Experiment configuration

### Data specification

openXDATA requires feature files to be saved in the ARFF format. The ARFF format originates from the Weka open-source data mining tool (https://waikato.github.io/weka-wiki/formats_and_processing/arff/). In short, ARFF files are essentially CSV files with a header section containing the attribute names and types.

An example can be found in `example/test1.arff`:

    @relation database1_arff

    @attribute instance_name string
    @attribute feature1 numeric
    @attribute feature2 numeric
    @attribute feature3 numeric
    @attribute classification_task1 { yes, no }
    @attribute regression_task1 numeric

    @data
    'inst1',2.0,3.0,4.0,yes,1.0
    'inst2',3.0,2.0,1.0,no,2.0
    'inst3',4.0,1.0,-1.1,yes,1.5
    'inst4',1.0,1.0,0.0,no,2.5
    'inst5',0.0,0.0,1.0,?,1.1

Note that the first string attribute in the ARFF file is assumed be to be an instance name and will be ignored for the purpose of machine learning, but will be used when writing out the data with completed labels.
If there is no such attribute (e.g. in `example/test2.arff`), openXDATA will internally generate one, using the name of the ARFF file and numbering the instances.

openXDATA requires at least a specification of the "cross-labeling" set, which is composed of several partially labeled datasets.
The specification contains the names of the ARFF files and the number of target attributes in each file, in CSV format.
An example specification can be found in `example/cl.spec`:

    test1.arff,2
    test2.arff,3
    test_unlab.arff,0

In this example, the file `test1.arff` contains two labels (`classification_task1` and `regression_task1`, cf. above), the file `test2.arff` contains three labels and the file `test_unlab.arff` is completely unlabeled (it only contains features).
The features (number of features, names and types) need to match in all the ARFF files, otherwise an error is thrown.
Optionally, one can also provide an evaluation specification. In this case, openXDATA will perform a test set evaluation after each iteration of the CDLC algorithm.

Features standardized ...

### Options

#### General options

Name | Description | Type | Default value
---- | ----------- | ---- | -------------
cl_spec | Filename of the cross-labeling specification (cf. above) | String | None
eval_spec | Filename of the evaluation specification (cf. above) | String | None
display_config | Whether to display the configuration variables on startup (including default ones) | Boolean | True
display_tasks | Whether to display the names of the tasks (target attributes) and the type (classification/regression) | Boolean | True
random_seed | Random seed (TensorFlow, NumPy and Python) | Integer | 42
standardize_features_per_datafile | Whether the features should be standardized (to zero mean and unit variance) per each data file (line in the cross-labeling/evaluation specification) | Boolean | True
standardize_features_global | Whether the features should be standardized (to zero mean and unit variance) globally | Boolean | False
standardize_test_features_on_train | Whether the features of test data should be standardized using the parameters of the training data | Boolean | True
standardize_labels | Whether regression labels should be standardized (to zero mean and unit variance) per target attribute (task) | Boolean | True

#### Training options

Name | Description | Type | Default value
---- | ----------- | ---- | -------------
batch_size | Batch size (in number of instances) for training | Integer | 32
num_epochs | Number of epochs to train the initial multi-task DNN (on the labeled data) | Integer | 10
learning_rate | Learning rate for training | Float | 0.01
learning_rate_decay | Learning rate decay for training (after N batches, the learning rate is scaled by 1/(1+decay*N)) | Float | 0.02
dropout_rate | Dropout for training | Float | 0.1
l2_regularization | L2 regularization weight | Float | 0.0001
use_dropout_for_eval | Whether to use the average of multiple dropout passes as predicted labels in evaluation | Boolean | True

#### Options for the multi-task DNN

Name | Description | Type | Default value
---- | ----------- | ---- | -------------
hidden_sizes | The hidden sizes for the multi-task DNN | List of integers | `[1024, 1024, 1024]`
num_shared_hidden_layers | The number of shared hidden layers in the multi-task DNN (all remaining hidden layers will be specific for each task) | Integer | 2

#### Cross-labeling (CDLC) options

Name | Description | Type | Default value
---- | ----------- | ---- | -------------
num_dropout_passes | Number of dropout passes for predicting labels | Integer | 10
num_cl_iters | Number of cross-labeling iterations | Integer | 2
cl_inst_per_iter | Number of instances to label per iteration | Integer | 200
cl_retrain_from_scratch | Whether to retrain a network from random initialization in each CDLC iteration | Boolean | False
cl_retrain_num_epochs | Number of epochs to retrain the network in each CDLC iteration | Integer | 1
cl_randomize_selection | Whether to label a random selection of instances (True) or use confidences for selecting instances in each iteration (False) | Boolean | False
final_dataset_name | The name of the dataset after cross-labeling (ARFF file and relation name) | String | `cross_labeled_dataset`
standardize_predictions | Whether to standardize (to zero mean and unit variance) the model predictions in each CDLC iteration | Boolean | False

### Real-world example

We include a real-world example from speech emotion recognition with the openXDATA tool. It can be found in the `example_emotion` folder.
The training data consists of four data files with 4,290 instances and 216 acoustic features per instance, and the test data contains 1,241 instances.
More details about how the data were obtained from the IEMOCAP corpus of emotional speech can be found in the paper referenced above.
The real-world example can be run by

    cd example_emotion
    python ../main.py IEMOCAP.yaml

After completion, the file `IEMOCAP_cross_labeled.arff` contains the cross-labeling set with completed labels.
Moreover, the UAR / CC performance is printed after each CDLC iteration.
