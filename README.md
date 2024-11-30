# Misogyny-Post-detection

### ENVIRONMENT CONFIGURATION 

Requirements:

To replicate the experiments,  environments are required.

```conda create --name <env> --file environments/requirements_fusion.txt```

#### DATASET DOWNLOAD
For the MAMI dataset, The data may be distributed upon request and for academic purposes only. To request the datasets, please fill out the following form: https://forms.gle/AGWMiGicBHiQx4q98
After submitting the required info, participants will have a link to a folder containing the datasets in a zip format (train, training and development) and the password to uncompress the files.

#### UTILS

The utils folder contains the code to process the data from the MAMI dataset to be easily used for training, validation and testing.
The dataset folder already contains the JSON files for the train, test and validation set.
Download the images from the dataset and place it in the 'dataset' directory

#### PRE-RUN

Our model will be provided upon request for testing.
After downloading the model and save in the "saved" folder
```
└───saved
     └───mami_random98.pth
└───dataloader_adv_train.py
└───...
└───robresGAT_MAMI.py
└───...
```

#### RUN

Run the robresGAT_MAMI.py file to execute the project

#### RESULTS

The preds folder contains the predictions of our model on the test set for human level judgment.
The report folder contains the report of training the model as well as its perfomance on the validation dataset.

#### DOI Link
https://doi.org/10.5281/zenodo.14246012