import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import * #To import all the functions developped in this internship

#Import the data file
#Path file in Windows
file_path = 'D:\Documentos pc Acer\Descargas pc Acer\ETIS\dataFiles\meteor_003019_long.csv' # Modify according to the file path
#Path file in Ubuntu
#file_path = ('/users/Downloads/meteor_003019_long.csv') # Modify according to the file path

with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)#Originally, dtype=float, I changed to int because there was an error trying
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.

#Size of the image/matrix
num_pixels_x = max(events[ :, 0])
num_pixels_y = max(events[ :, 1])

#Total time of the data
time_data = events[-1, -1]

#Create the matrices for events
positive_events_matrix = np.zeros((num_pixels_y + 1, num_pixels_x + 1))
negative_events_matrix = np.zeros((num_pixels_y + 1, num_pixels_x + 1))

#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    if (events[i,2] == 1): #if polarity = 1
        counting_events_per_pixel(positive_events_matrix, events[i,0], events[i,1])

    elif (events[i,2] == 0): #if polarity = 0
        counting_events_per_pixel(negative_events_matrix, events[i,0], events[i,1])

display_4_matrices(positive_events_matrix, negative_events_matrix, file_path)
