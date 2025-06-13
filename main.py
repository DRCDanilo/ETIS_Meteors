import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from events_display import *

#######################################################################################################################
#Import the data file
file_path = ('/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_003019_long.csv') # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor.csv' # Modify according to the file path

with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)#Originally, dtype=float, I changed to int because there was an error trying
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.

#Filter the file by time: AS IT IS LONGER THANT THE REFERENCE DATA (file meteor.csv), IT IS NECESSARY TO HAVE THE SAME TIME INTERVAL
if ( events[-1,-1] > 1461730 ):
    mask = events[:, -1] <= 1461730
    events = events[mask]
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.
    print('Already filered by time')
    print('The first event is : ', events[0])
    print('The last event is : ', events[-1])
    print('The total time is ' , events[-1,-1]-events[0,-1])

#Size of the image/matrix
numPixelsX = max(events[ :, 0])
numPixelsY = max(events[ :, 1])
print('The number of pixels in x is', numPixelsX + 1)
print('The number of pixels in y is', numPixelsY + 1)

#Matrix for events
PositiveEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))
NegativeEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))

#Array to count the events per pixel. nx5 = [xCoord, yCoord, positiveEvents, negativeEvents, totalEvents]
#The idea is to replace the function eventsPerPixel() and the two variables histoPositiveEvents, histoNegativeEvents to have just one
#varibale with both positive and negative events
pixelsEvents = np.zeros([1, 5], dtype = int)

# Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    makePixelsHistogram(events[i, 0], events[i, 1], events[i, 2])

    if (events[i, 2] == 1):
        CountingEventsPerPixel(PositiveEventsMatrix, events[i, 0], events[i, 1])

    elif (events[i, 2] == 0):
        CountingEventsPerPixel(NegativeEventsMatrix, events[i, 0], events[i, 1])

#Delete the first row of pixelsEvents because is 0,0,0,0,0
#The idea is to replace the function eventsPerPixel() and the two variables histoPositiveEvents, histoNegativeEvents to have just one
#varibale with both positive and negative events
pixelsEvents = np.delete(pixelsEvents, 0, 0)

#######################################################################################################################

# display4Matrix()
displayZoneHistogram(20000, 600000, 391, 438, 3, 'Background')
displayZoneBihistogram(20000, 600000, 391, 438, 3, 'Background')
plt.show()