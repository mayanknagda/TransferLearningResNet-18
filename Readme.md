# Pretrained ResNet-18

## Install the env files through conda using requirements.yml (conda env create --file requirements.yml)

## main.py

### To run, simply use main.py. Parameters used in the experiments are available in the Namespace at line 43. To give separate parameters uncomment line at 42 and comment line 43. Or edit the Namespace and run it. use data_location argument in the specified format to load the torchvision data.

### The output_dir argument stores the output with the dir name in the output/ directory.

### config_type argument uses types of config as input. 5 seperate configs are provided in the model (a, b, c, d, e) as it is given in the test task.

### Rest of arguments are self explanatory

### remember to delete the run in the output/ dir if you are generating run with the same name. since 5 separate runs are already present. change naming to 'config_a_run2' style if you want to run the code and still keep the already given runs in the output/ dir.

## train.py

### used to write the train logic

## model.py

### where the model is located

## utils.py

### utility functions are present here

## analysis.ipynb

### jupyter notebook used for analysis purposes. It takes input from the output/ dir automatically and does the processing. Remember to save the output (name of data_location argument in main.py) which starts with 'config' to account in the analysis.

## analysis.html

### HTML version of jupyternotebook with analysis of runs in output directory.
