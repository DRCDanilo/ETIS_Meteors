import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import *

#################################################################################################################################################
#Main Program
#################################################################################################################################################

#Import the data file
# ___________ FOR METEOR 00:29:40 _______________________
file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor.csv' # Modify according to the file path
#file_path = ('/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_003019_long.csv') # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/stars_0037.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_235935.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_232112_big.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_233126.csv' # Modify according to the file path





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




#p = np.where( ( pixelsEvents[:,0] == 236 ) & ( pixelsEvents[:,1] == 289) )
#pixel1 = pixelsEvents[p]
#print(pixel1)






# displayZoneHistogram(20000, 600000, 284, 59, 3, 'Capella')
# displayZoneBihistogram(20000, 600000, 284, 59, 3, 'Capella')
# plt.show()

###############################################
#TEST FOR THE NUMBER OF EVENTS PER UNIT OF TIME
###############################################

#Filtering to just have the 6 stars
unitOfTime = 1000000 # Parameter
pixelsEvents = addEventsByTime(pixelsEvents, timeData, unitOfTime)
remainPixels = directNeighbors(pixelsEvents, 0.13, 4, 8,5)
remainPixels = filterArray(remainPixels, 2, 4, 1)

remainPixels = filterArray(remainPixels, 7, 4, 2)
#remainPixels = isStar(remainPixels) #The 4 stars
remainPixels = remainPixels.astype(int)
print('There are ',len(remainPixels), 'pixels.')
print(remainPixels)

finalMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))

displayPixels(remainPixels, finalMatrix, file_path)


#display4Matrix(PositiveEventsMatrix, NegativeEventsMatrix, file_path)


# #Find the 1st star
# p = np.where( ( pixelsEvents[:,0] == 384 ) & ( pixelsEvents[:,1] == 426) )
# #Array with just the stars
# justStars = pixelsEvents[p]
# #Find the 2nd star
# p = np.where( ( pixelsEvents[:,0] == 534 ) & ( pixelsEvents[:,1] == 306) )
# #Add the 2nd star
# justStars = np.vstack((justStars, pixelsEvents[p]))
# #Find the 3rd star
# p = np.where( ( pixelsEvents[:,0] == 412 ) & ( pixelsEvents[:,1] == 216) )
# #Add the 3rd star
# justStars = np.vstack((justStars, pixelsEvents[p]))
# #Find the 4th star
# p = np.where( ( pixelsEvents[:,0] == 236 ) & ( pixelsEvents[:,1] == 289) )
# #Add the 4th star
# justStars = np.vstack((justStars, pixelsEvents[p]))
#
# justStars = justStars.astype(int)
# print(justStars)
#
# suma = PositiveEventsMatrix + NegativeEventsMatrix #The total number of events in all the data
#
# maskMatrix = np.empty((numPixelsY + 1, numPixelsX + 1))
# maskMatrix.fill(np.nan) #Create a matrix with NaN terms
# maskMatrix = onesMask(justStars, maskMatrix) #Create the matrix with ones in the neighbors pixels and NaN terms
#
# neighborsMatrix = suma * maskMatrix #Keep just te neighbors without background, stars and meteor
#
# justNaN = np.isnan(neighborsMatrix)
# justNaN = np.logical_not(justNaN) #Mask to remove NaN elements
#
# onlyNeighbors = neighborsMatrix[justNaN] #Keep just the neighbors without NaN elements
#
# getParameters(onlyNeighbors)
#
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
#
# pos1 = ax1.imshow(maskMatrix, cmap = 'cividis_r', interpolation = 'none')
# fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
# ax1.set_title("Neighbors Pixels' Mask Matrix")
# ax1.set_xlabel('pixels')
# ax1.set_ylabel('pixels')
# displayExtraInfo(ax1, file_path)
#
# pos2 = ax2.imshow(suma, cmap = 'cividis_r', interpolation = 'none', vmax = 25)
# fig2.colorbar(pos2, ax=ax2, shrink = 0.8)
# ax2.set_title('Total Events Matrix')
# ax2.set_xlabel('pixels')
# ax2.set_ylabel('pixels')
# displayExtraInfo(ax2, file_path)
#
# pos3 = ax3.imshow(neighborsMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = 25)
# fig3.colorbar(pos3, ax=ax3, shrink = 0.8)
# ax3.set_title("Neighbors Pixels' Matrix")
# ax3.set_xlabel('pixels')
# ax3.set_ylabel('pixels')
# displayExtraInfo(ax3, file_path)
#
# plt.show()