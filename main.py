import csv #To read the data in the .csv file
import numpy as np #To work with numpy
from matplotlib import pyplot as plt #To work with Matplotlib and display images and figures
from events_display import * #To import all the functions developped in this internship

#################################################################################################################################################
#Main Program
#################################################################################################################################################

#Import the data file
# ___________ FOR METEOR 00:29:40 _______________________

file_path = 'D:\Documentos pc Acer\Descargas pc Acer\ETIS\dataFiles\meteor.csv' # Modify according to the file path
#file_path = ('/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_003019_long.csv') # Modify according to the file path

#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/stars_0037.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_235935.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_232112_big.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_233126.csv' # Modify according to the file path




#start = time.time()#To count the time to read the file. Delete this
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
pixelsEvents = np.zeros([1, 5], dtype = int)

#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):

    pixelsEvents = makePixelsHistogram(events[i,0], events[i,1], events[i,2], pixelsEvents)

    if (events[i,2] == 1):
        counting_events_per_pixel(PositiveEventsMatrix, events[i,0], events[i,1])

    elif (events[i,2] == 0):
        counting_events_per_pixel(NegativeEventsMatrix, events[i,0], events[i,1])



#Delete the first row of pixelsEvents because is 0,0,0,0,0
pixelsEvents = np.delete(pixelsEvents, 0, 0)

#Add a column for the number of events per time unit (i.e. events x second)
pixelsEvents = np.hstack( ( pixelsEvents , np.zeros( [len(pixelsEvents), 1], dtype = float ) ) )  #Add the pixel to the array


###############################################
#TEST FOR THE NUMBER OF EVENTS PER UNIT OF TIME
###############################################

#Filtering to just have the 6 stars
# unitOfTime = 1000000 # Parameter to define the unit of time of the events
# pixelsEvents = addEventsByTime(pixelsEvents, timeData, unitOfTime) #Add the number of events per unit of time to every pixel
# remainPixels = directNeighbors(pixelsEvents, 0.6, 3, 8, 5) #Filtering by direct neighbors
# remainPixels = filterArray(remainPixels, 20, 4, 1) #Filtering by number of events per unit of time
# remainPixels = isStar(remainPixels) #Identify is one or more pixels belong to the same star
# remainPixels = remainPixels.astype(int)
# print('There are ',len(remainPixels), 'pixels.')
# print(remainPixels)
#
# #finalMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))
#
# #displayPixels(remainPixels, finalMatrix, file_path)
#
#
# suma = PositiveEventsMatrix + NegativeEventsMatrix #The total number of events in all the data
# #Matrix to make the mask
# mask = np.ones((numPixelsY + 1, numPixelsX + 1))
# mask = NaNMask(remainPixels, mask) #Mask : NaN terms in stars and meteor, 1 the rest of the pixels
# mask = suma * mask #Keep just te background without stars and meteor
# justNaN = np.isnan(mask)
# justNaN = np.logical_not(justNaN) #Mask to remove NaN elements
# background = mask[justNaN] #Keep just the background without NaN elements
# getParameters(background) #Get the parameters of the background
#
# print('The real background information is :') #Analysis of the real background
# #maskRealBackground = background != 0
# maskZeroEvents = background == 0 #Mask with only the pixels with zero events
# maskRealBackground = np.logical_not(maskZeroEvents)
# realBackground  = background[maskRealBackground] #Get the real background
# getParameters(realBackground) #Get the parameters of the real background
#
# #Make NaN the pixels with zero events in the mask matrix
#
#
#
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
#
# pos1 = ax1.imshow(suma, cmap = 'cividis_r', interpolation = 'none')
# fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
# ax1.set_title("Total Events Matrix")
# ax1.set_xlabel('pixels')
# ax1.set_ylabel('pixels')
# displayExtraInfo(ax1, file_path)
#
# pos2 = ax2.imshow(mask, cmap = 'cividis_r', interpolation = 'none', vmax = 25)
# fig2.colorbar(pos2, ax=ax2, shrink = 0.8)
# ax2.set_title('Mask Matrix')
# ax2.set_xlabel('pixels')
# ax2.set_ylabel('pixels')
# displayExtraInfo(ax2, file_path)
#
# plt.show()





















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
# pos1 = ax1.imshow(mask, cmap = 'cividis_r', interpolation = 'none')
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
# pos3 = ax3.imshow(suma, cmap = 'cividis_r', interpolation = 'none', vmax = 25)
# fig3.colorbar(pos3, ax=ax3, shrink = 0.8)
# ax3.set_title("Neighbors Pixels' Matrix")
# ax3.set_xlabel('pixels')
# ax3.set_ylabel('pixels')
# displayExtraInfo(ax3, file_path)
#
# plt.show()
