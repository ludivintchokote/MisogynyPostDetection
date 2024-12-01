# Misogyny-Post-detection

### ENVIRONMENT CONFIGURATION 

Requirements:

To replicate the experiments,  environments are required.

```conda create --name <env> --file environments/requirements_fusion.txt```

#### DATASET DOWNLOAD
For the MAMI dataset, The data may be distributed upon request and for academic purposes only. To request the datasets, please fill out the following form: https://forms.gle/AGWMiGicBHiQx4q98
After submitting the required info, participants will have a link to a folder containing the datasets in a zip format (train, training and development) and the password to uncompress the files.
After downloading the images, organize it as follows:

```
└───datasets
     └───mami
         └───files
         └───img
└───models
    └───roberta_resnet
        └───saved
        └───Preds
        └───dataloader_adv_train.py
        └───robresGAT_MAMI.py
        └───testing.py
```

#### UTILS

The utils folder contains the code to process the data from the MAMI dataset to be easily used for training, validation and testing.
The dataset folder already contains the JSON files for the train, test and validation set.
Download the images from the dataset and place it in the 'dataset' directory.


#### PRE-RUN

Our model will be provided upon request for testing and evaluation.
After downloading the model,save it in the "saved" folder following the file structure below
```
└───saved
     └───mami_random98.pth
└───dataloader_adv_train.py
└───Preds
└───robresGAT_MAMI.py
└───testing.py
```

#### RUN

Run the robresGAT_MAMI.py file to execute the project.
Run the testing.py file to test the model in the saved path.
Run the make_jsons_mami.py to create json files for the train, validation and test set of the dataset.

#### RESULTS

The preds folder contains the predictions of our model on the test set for human level judgment.
The report folder contains the report of training the model as well as its perfomance on the validation dataset.
