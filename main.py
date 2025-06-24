import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import *

#################################################################################################################################################
#Main Program
#################################################################################################################################################

#Import the data file
# ___________ FOR METEOR 00:29:40 _______________________
#file_path = ('/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_003019_long.csv') # Modify according to the file path
file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor.csv' # Modify according to the file path


#Message text
##dataName = file_path[55:-4]
##actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
##fileImgName = dataName + '_' + actualDataTime


#start = time.time()#To count the time to read the file. Delete this
with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)#Originally, dtype=float, I changed to int because there was an error trying
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.


#CODE TO TEST FILTERING. AS IT IS LONGER THANT THE REFERENCE DATA, IT IS NECESSARY TO HAVE THE SAME TIME INTERVAL
#filter the file by time
# if ( events[-1,-1] > 1461730 ) :
#     mask = events[:, -1] <= 1461730
#     events = events[mask]
#     events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.
#     print('Already filered by time')
#     print('The first event is : ', events[0])
#     print('The last event is : ', events[-1])
#     print('The total time is ' , events[-1,-1]-events[0,-1])


#Size of the image/matrix
numPixelsX = max(events[ :, 0])
numPixelsY = max(events[ :, 1])
print('The number of pixels in x is', numPixelsX + 1)
print('The number of pixels in y is', numPixelsY + 1)

#Total time of the data
timeData = events[-1, -1]
print('The total time is ' , events[-1, -1])


#Matrix for events
PositiveEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))
NegativeEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))


#Array to count the events per pixel. nx5 = [xCoord, yCoord, positiveEvents, negativeEvents, totalEvents]
pixelsEvents = np.zeros([1, 5], dtype = int)

#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    pixelsEvents = makePixelsHistogram(events[i,0], events[i,1], events[i,2], pixelsEvents)

    if (events[i,2] == 1):
        CountingEventsPerPixel(PositiveEventsMatrix, events[i,0], events[i,1])

    elif (events[i,2] == 0):
        CountingEventsPerPixel(NegativeEventsMatrix, events[i,0], events[i,1])



#Delete the first row of pixelsEvents because is 0,0,0,0,0
pixelsEvents = np.delete(pixelsEvents, 0, 0)

#Add a column for the number of events per time unit (i.e. events x second)
pixelsEvents = np.hstack( ( pixelsEvents , np.zeros( [len(pixelsEvents), 1], dtype = float ) ) )  #Add the pixel to the array


# # #Test to filter just the stars and find out if the method works.
# remainPixels = directNeighbors(pixelsEvents, 1, 3, 8)
# remainPixels = filterArray(remainPixels, 5, 3, 1)
# mean = np.ceil(np.mean(remainPixels[:, -1]))
# print('The average number of events in the array is : ', mean)
# remainPixels = filterArray(remainPixels, mean, 3, 1)
# remainPixels = isStar(remainPixels)
# print('There are ', len(remainPixels), 'possible stars')
# print('The possible stars are: \n', remainPixels)
# displayPixels(remainPixels)




# displayZoneHistogram(20000, 600000, 284, 59, 3, 'Capella')
# displayZoneBihistogram(20000, 600000, 284, 59, 3, 'Capella')
# plt.show()

###############################################
#TEST FOR THE NUMBER OF EVENTS PER UNIT OF TIME
###############################################


unitOfTime = 1000000 # Parameter

pixelsEvents = addEventsByTime(pixelsEvents, timeData, unitOfTime)
remainPixels = directNeighbors(pixelsEvents, 0.6, 3, 8,5)
remainPixels = filterArray(remainPixels, 20, 4, 1)
remainPixels = isStar(remainPixels)
remainPixels = remainPixels.astype(int)
print('quedan :',len(remainPixels))
print(remainPixels)


suma = PositiveEventsMatrix + NegativeEventsMatrix


mask = np.ones((numPixelsY + 1, numPixelsX + 1))
mask = NaNMask(remainPixels, mask)

mask = mask * suma

justNaN = np.isnan(mask)
justNaN = np.logical_not(justNaN)

background = mask[justNaN]

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

pos1 = ax1.imshow(suma, cmap='cividis_r', interpolation='none', vmax = 35)
fig1.colorbar(pos1, ax=ax1, shrink=0.8)  # Colorbar
ax1.set_title('Sum Events Matrix')
ax1.set_xlabel('pixels')
ax1.set_ylabel('pixels')
displayExtraInfo(ax1, file_path)

pos2 = ax2.imshow(mask, cmap='cividis_r', interpolation='none', vmax = 35)
fig2.colorbar(pos2, ax=ax2, shrink=0.8)  # Colorbar
ax2.set_title('Mask Matrix')
ax2.set_xlabel('pixels')
ax2.set_ylabel('pixels')
displayExtraInfo(ax2, file_path)

plt.show()



###############################################
#TEST FOR THE MASK NaN FOR THE BACKGROUND
###############################################
# mask = np.ones((1, 3))
# mask[0,0] = np.nan
# print('The mask is : ', mask)
# numbers = [2, 4, 6]
# print('The the nulbers are : ', numbers)
# r = mask * numbers
# print('The r is : ', r)
#
# rnan = np.isnan(r)
# rnan = np.logical_not(rnan)
# print('The r is : ', rnan)
#
# rfinal = r[rnan]
# print('The r is : ', rfinal)