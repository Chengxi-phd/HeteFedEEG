# HeteFedEEG: Prototype-Transfer Heterogeneous Federated Learning for EEG-based Emotion Recognition

## Prepare dataset
We use two publicly available datasets: SEED, DEAP. 
SEED can be requested at https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/. 
DEAP can be requested at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/.

Feel free to write your own code to preprocess and split each dataset to training and test sets.
Replace the preprocess module in main.py.

## Train model
Orgnize your datasets in a folder and set the path as follow:
```
dataset_dir = "/yourDatasetPath/"
```
Default values of parameters have been set in main.py. You can run with default with following simple comment:

```
python main.py
```
