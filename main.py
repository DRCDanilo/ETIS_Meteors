import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import * #To import all the functions developped in this internship

#Import the data file
#Path file in Windows
file_path = 'D:\Documentos pc Acer\Descargas pc Acer\ETIS\dataFiles\meteor.csv' # Modify according to the file path
#Path file in Ubuntu
#file_path = ('/users/Downloads/meteor_003019_long.csv') # Modify according to the file path

with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.

#Size of the image/matrix
num_pixels_x = max(events[ :, 0])
num_pixels_y = max(events[ :, 1])

#Total time of the data
time_data = events[-1, -1]

#Array to count the events per pixel. nx5 = [x coord, y coord, positive events, negative events, total events]
pixel_events = np.zeros([1, 5], dtype = int)

###Loop through data array to create the pixel_events variable
##for i in range( len( events ) ):
##
##    pixel_events = count_pixel_events(events[i,0], events[i,1], events[i,2], pixel_events)
##
###Delete the first row of pixel_events because is 0,0,0,0,0
##pixel_events = np.delete(pixel_events, 0, 0)
##
###Add a column to pixel_events for the number of events per time unit (e.g. events/second)
##pixel_events = np.hstack( ( pixel_events , np.zeros([len( pixel_events ), 1], dtype = float ) ) )
##
###Filtering process
###Parameter to have events/second. 1000000 us = 1 s
##unit_of_time = 1000000
##
###Add the number of events/second to every pixel
##pixel_events = add_events_per_time(pixel_events, time_data, unit_of_time)
##
###Filtering by neighbors of the pxiels
##remain_pixels = direct_neighbors(pixel_events, 0.13, 4, 8, 5)
##
###Filtering by number of events/second
##remain_pixels = filter_array(remain_pixels, 2, 4, 1)
##
###Filtering by number of events/second
##remain_pixels = filter_array(remain_pixels, 7, 4, 2)
##
##remain_pixels = remain_pixels.astype(int)
##
###Print the pixels identified as stars
##print('There are',len(remain_pixels), ' pixels after filtering process')
##print('The pixels of the meteors are: ')
##print(remain_pixels)
##
###Display the output image with the pixels identified as stars
##output_image = np.zeros( ( num_pixels_y + 1, num_pixels_x + 1 ) )
##image_title = 'Meteors After Filtering Process'
##display_pixels(remain_pixels, output_image, file_path, image_title)

zone_double_histogram(events, 20000, 1000000, 391, 438, 3, 'Procyon', file_path)
