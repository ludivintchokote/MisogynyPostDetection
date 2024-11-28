import json
import pandas as pd

# Read the csv file
data = pd.read_csv('C:/Users/user/Desktop/Project/datasets/mami/training.csv', sep='\t', quotechar='"')

# Initialize an empty list to store the data
list_data = []
print("Column names:")
print(data.columns)

# Iterate through each row in the dataframe
for _, point in data.iterrows():
    # Create a new dictionary for each row
    dict_temp = {}

    # Add the image filename to the dictionary
    dict_temp['img'] = point['file_name']

    # Add the label to the dictionary
    # If the 'misogynous' column value is 1, set the label to 1
    # Otherwise, set the label to 0
    if point['misogynous'] == 1:
        dict_temp['label'] = 1
    else:
        dict_temp['label'] = 0

    # Add the text transcription to the dictionary
    dict_temp['text'] = point['Text Transcription']

    # Append the dictionary to the list
    list_data.append(dict_temp)

# Convert the list of dictionaries to a JSON object
json_object = json.dumps(list_data, indent=4)

# Write the JSON object to a file
with open("C:/Users/user/Desktop/Project/datasets/mami/files/train.json", "w") as outfile:
    outfile.write(json_object)
