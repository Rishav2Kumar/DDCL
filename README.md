# Data Distribution-based Curriculum Learning (DDCL)

Copyright &copy; 2025, ECOLS - All rights reserved.

**2025-01-30: Initial upload featuring Curriculum Learning and Self-Paced Learning.**

## 1. Static DDCL
*Data Distribution-based Curriculum Learning (DDCL)* or *Static DDCL* uses the data distribution of a dataset to build a curriculum based on the order of samples. It includes two types of scoring methods known as **DDCL (Density)** and **DDCL (Point)** which determine the difficulty of training samples and thus their order among other samples during training. **DDCL (Density)** uses the sample density to assign scores while **DDCL (Point)** utilises the Euclidean distance for scoring.

The Static DDCL approach was developed by Shonal Chaudhry with guidance from Anurag Sharma while being a member of the ECOLS research group.

**Paper**: S. Chaudhry and A. Sharma, *Data Distribution-Based Curriculum Learning*, IEEE Access, vol. 12, pp. 138429–138440, 2024, doi: 10.1109/ACCESS.2024.3465793.

## 2. Dynamic DDCL
*Dynamic DDCL* extends DDCL by adding a dynamically generated curriculum through self-paced learning. It adds two additional scoring methods known as **Self-Paced DDCL (Density)** and **Self-Paced DDCL (Point)** which use a combination of curriculum learning and self-paced learning.

The Dynamic DDCL approach was developed by Shonal Chaudhry with guidance from Anurag Sharma while being a member of the ECOLS research group.

**Paper**: S. Chaudhry and A. Sharma, *Dynamic Data Distribution-based Curriculum Learning*, Information Sciences, vol. 702, p. 121924, 2025, doi: (https://doi.org/10.1016/j.ins.2025.121924).

## Dependencies
- Python 3.12.7
- Tensorflow 2.17.0
- Pandas 2.2.2
- Scikit-Learn 1.5.1
- Imbalanced-Learn 0.12.3
- Matplotlib 3.9.2
- Seaborn 0.13.2

## Running the code
- Run the code through the `main.py` file. Program behaviour can be altered by changing the options in `config.ini`.
- There are two sub-program options, `DDCL` and `Plot`, which run the DDCL code or generate plots using output files obtained from the `DDCL` sub-program respectively.

## DDCL Examples
Some examples for running a specific `DDCL` configuration:

### Single model - train and predict
1. In `config.ini`, select one of these `learning_model` options: **Neural Network**, **SVM** or **Random Forest**.
2. Select a `training_strategy` option.
3. Use **train_and_predict** for the `nn_task` option.

### Train and save a model
1. In `config.ini`, select one of the `learning_model` options: **Neural Network**, **SVM**, **Random Forest**, **DDCL Ensemble** or **Standard Ensemble**.
2. Select a `training_strategy` option. Note: selecting **DDCL Ensemble** in Step 1 will use all options.
3. Use **train_and_save** for the `nn_task` option.

### Predict with ensemble models
1. In `config.ini`, select either **DDCL Ensemble** or **Standard Ensemble** for the `learning_model` option.
    - For **DDCL Ensemble**, the `training_strategy` option is ignored.
    - For **Standard Ensemble**, select a `training_strategy` option that has a saved model available.
2. Use **load_and_predict** for the `nn_task` option.

