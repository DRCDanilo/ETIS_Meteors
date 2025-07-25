import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import * #To import all the functions developped in this internship

#################################################################################################################################################
#Main Program
#################################################################################################################################################

#Import the data file
#Path file in Windows
file_path = 'D:\Documentos pc Acer\Descargas pc Acer\ETIS\dataFiles\meteor.csv' # Modify according to the file path
#Path file in Ubuntu
#file_path = ('/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_003019_long.csv') # Modify according to the file path

with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)#Originally, dtype=float, I changed to int because there was an error trying
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.


#Size of the image/matrix
numPixelsX = max(events[ :, 0])
numPixelsY = max(events[ :, 1])

#Total time of the data
timeData = events[-1, -1]

#Create the matrices for events
PositiveEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))
NegativeEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))


#Array to count the events per pixel. nx5 = [xCoord, yCoord, positiveEvents, negativeEvents, totalEvents]
#pixelsEvents = np.zeros([1, 5], dtype = int)

#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    #pixelsEvents = makePixelsHistogram(events[i,0], events[i,1], events[i,2], pixelsEvents)

    if (events[i,2] == 1):
        counting_events_per_pixel(PositiveEventsMatrix, events[i,0], events[i,1])

    elif (events[i,2] == 0):
        counting_events_per_pixel(NegativeEventsMatrix, events[i,0], events[i,1])


#Delete the first row of pixelsEvents because is 0,0,0,0,0
#pixelsEvents = np.delete(pixelsEvents, 0, 0)

#Add a column for the number of events per time unit (i.e. events x second)
#pixelsEvents = np.hstack( ( pixelsEvents , np.zeros( [len(pixelsEvents), 1], dtype = float ) ) )  #Add the pixel to the array

display_4_matrices(PositiveEventsMatrix, NegativeEventsMatrix, file_path)
