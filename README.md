# Misogyny-Post-detection

This is the official repo of our paper entitled 

### Detecting Misogynistic Posts on Social Media using a Robust Multimodal Framework: Integrating ResNet, RoBERTa, and Graph Attention Networks 

Requirements:

To replicate the experiments,  environments are required.

```conda create --name <env> --file environments/requirements_fusion.txt```

#### DATASET
For the MAMI dataset, The data may be distributed upon request and for academic purposes only. To request the datasets, please fill out the following form: https://forms.gle/AGWMiGicBHiQx4q98

After submitting the required info, participants will have a link to a folder containing the datasets in a zip format (train, training and development) and the password to uncompress the files.

#### UTILS

The utils folder contains the code to process the data from the MAMI dataset to be easily used for training, validation and testing.

#### RESULTS

The preds folder contains the predictions of our model on the test set for human level judgment.
The report folder contains the report of training the model as well as its perfomance on the validation dataset.

#### DOI Link
https://doi.org/10.5281/zenodo.14246012