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

#Array to count the events per pixel. nx5 = [xCoord, yCoord, positiveEvents, negativeEvents, totalEvents]
pixelsEvents = np.zeros([1, 5], dtype = int)

#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    pixelsEvents = count_pixel_events(events[i,0], events[i,1], events[i,2], pixelsEvents)


#Delete the first row of pixelsEvents because is 0,0,0,0,0
pixelsEvents = np.delete(pixelsEvents, 0, 0)

#Add a column for the number of events per time unit (i.e. events x second)
pixelsEvents = np.hstack( ( pixelsEvents , np.zeros( [len(pixelsEvents), 1], dtype = float ) ) )  #Add the pixel to the array

#Filtering to just have the 6 stars
unitOfTime = 1000000 # Parameter to define the unit of time of the events
pixelsEvents = addEventsByTime(pixelsEvents, time_data, unitOfTime) #Add the number of events per unit of time to every pixel
remainPixels = directNeighbors(pixelsEvents, 0.6, 3, 8, 5) #Filtering by direct neighbors
remainPixels = filterArray(remainPixels, 20, 4, 1) #Filtering by number of events per unit of time
remainPixels = isStar(remainPixels) #Identify is one or more pixels belong to the same star
remainPixels = remainPixels.astype(int)
print('There are',len(remainPixels), 'stars:')
print(remainPixels)
