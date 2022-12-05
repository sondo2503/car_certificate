import pandas as pd

# read contents of csv file
headerList=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']
file = pd.read_csv("results.csv")
file.to_csv("results.csv", header=headerList, index=False)

# # display modified csv file
# file2 = pd.read_csv("resulsts.csv")
# print('\nModified file:')
# print(file2)