#################################################################################################################################################
#Imports
#################################################################################################################################################
#To read the data in the .csv file
import csv
import os

#To work with Matplotlib and display images and figures
from matplotlib import pyplot as plt
#To work with numpy
import numpy as np
#To know the date and time to save them with the images
from datetime import datetime
#To measure the time reading the CSV file
import time

#To have interactive plots in Ubuntu: see the coordinates of a pixel wih the cursor of the mouse
import matplotlib
matplotlib.use('Qt5Agg')


#To display an image
import PIL
from PIL import Image


#import sys
#sys.path.append("/")



#################################################################################################################################################
#Functions
#################################################################################################################################################
#Function to fill a matrix with the number of changes per pixel
def CountingEventsPerPixel (m, xCoord, yCoord):#m: matrix to fill. xCoord: coordinate x pixel. yCoord: coordinate y pixel.
    m[yCoord, xCoord] += 1


#Function to save the image
def saveImage ():
    dataName = file_path[56:-4]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_' + actualDataTime + '.png'

    save_folder = "/users/danidelr86/Téléchargements/ETIS_stars/images/22h19_meteor"

    full_path = os.path.join(save_folder, fileImgName)
    
    plt.savefig(full_path,bbox_inches='tight', pad_inches=0)

#Function to display additional information in the image
def displayExtraInfo (axe):
    dataName = file_path[56:]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fileImgName = dataName + '_' + actualDataTime
    axe.annotate('Data file: ' + dataName, xy = (0, -25), xycoords = 'axes points', fontsize = 8)
    axe.annotate('Date: ' + actualDataTime, xy = (0, -33), xycoords = 'axes points', fontsize = 8)

#Function to display the image's histogram
def displayHistogram():
    print('Function displayHistogram')
    #Histogram
    #Variable to define the bins of the histogram
    binWidth = 50000
    #Variable to know the time of the las event
    timeLastEvent = events[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)

    fig, ax = plt.subplots()
    ax.hist(events[:,-1], bins=xbins, edgecolor = 'orange')

    #plot the xdata locations on the x axis:
    ax.plot(events[:,-1], 0*events[:,2], 'd')

    plt.title("Histogram of image's events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    displayExtraInfo(ax)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
    plt.show()
    

#Function to display the image's bihistogram: histogram of the positive and negative events
def displayBihistogram():
    print('Function displayHistogram')
    #Histogram
    #Variable to define the bins of the histogram
    binWidth = 50000
    #Variable to know the time of the las event
    timeLastEvent = events[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)

    #BiHistogram - Histogram for positive and negative events
    fig, ax = plt.subplots()
    #Mask for positive events
    positiveMask = events[:,2] == 1
    #Mask for negative events
    negativeMask = events[:,2] == 0

    #Plot the histogram for positive events
    ax.hist(events[:,-1][positiveMask], bins=xbins, edgecolor = 'black', label = 'Positive Events')
    #Plot the histogram for negative events
    ax.hist(events[:,-1][negativeMask], weights = -np.ones_like(events[:,-1][negativeMask]), bins=xbins, edgecolor = 'black', label = 'Negative Events')

    #Plot the data (positive and negative) along the x axis
    #plot the xdata locations on the x axis:
    #ax.plot(events[:,-1][positiveMask], 0*events[:,-1][positiveMask], '+', c = 'w')
    #ax.plot(events[:,-1][negativeMask], 0*events[:,-1][negativeMask], 'o', c = 'k')

    plt.title("Histogram of image's events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    ax.legend()
    #Extra info image
    displayExtraInfo(ax)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
    plt.show()

#Function to display the 4 matrix of the image.
#Positive Events Matrix, Negative Events Matrix, Sum Events Matrix, Average Events Matrix
def display4Matrix():
    print('Function display4Matrix')

    #Average Matrix
    AverageMatrix = PositiveEventsMatrix - NegativeEventsMatrix

    #Sum Matrix
    SumMatrix = PositiveEventsMatrix + NegativeEventsMatrix


    #Display images
    #Display Positive Events Matrix
    #Variable: Max value of the matrix
    MaxPosMatrix = np.max(PositiveEventsMatrix)
    print('The maximun number of events in the positive matrix is: ', np.max(PositiveEventsMatrix))
    print('The minimun number of events in the positive matrix is: ', np.min(PositiveEventsMatrix, where = PositiveEventsMatrix > 0, initial = np.inf))


    #Display Negative Events Matrix
    #Variable: Max value of the matrix
    MaxNegMatrix = np.max(NegativeEventsMatrix)
    print('The maximun number of events in the negative matrix is: ', np.max(NegativeEventsMatrix))
    print('The minimun number of events in the negative matrix is: ', np.min(NegativeEventsMatrix, where = NegativeEventsMatrix > 0, initial = np.inf))


    #Display Sum Matrix
    #Variable: Max value of the matrix
    MaxSumMatrix = np.max(SumMatrix)
    print('The maximun number of events in the sum matrix is: ', np.max(SumMatrix))
    print('The minimun number of events in the sum matrix is: ', np.min(SumMatrix, where = SumMatrix > 0, initial = np.inf))


    #Display Average Matrix
    #Variable: Max value of the matrix
    MaxAvgMatrix = np.max(AverageMatrix)
    #Variable: Min value of the matrix
    MinAvgMatrix = np.min(AverageMatrix)
    print('The maximun number of events in the average matrix is: ', np.max(AverageMatrix))
    print('The minimun number of events in the average matrix is: ', np.min(AverageMatrix))

    #Transform the range values of AverageMatrix to 0-255 to display the image
    #AverageMatrix =        ( (255) / (np.max(AverageMatrix) - np.min(AverageMatrix)) )  * (AverageMatrix - np.min(AverageMatrix))      
    #print('The maximun number of events in the average matrix after transformation is: ', np.max(AverageMatrix))
    #print('The minimun number of events in the average matrix after transformation is: ', np.min(AverageMatrix))

    
    vMaxScale = 20

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()


    pos1 = ax1.imshow(PositiveEventsMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
    ax1.set_title('Positve Events Matrix')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')
    xyMax = np.where(PositiveEventsMatrix >= MaxPosMatrix)
    ax1.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax1)


    pos2 = ax2.imshow(NegativeEventsMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig2.colorbar(pos2, ax=ax2, shrink = 0.8)
    ax2.set_title('Negative Events Matrix')
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')
    xyMax = np.where(NegativeEventsMatrix >= MaxNegMatrix)
    ax2.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax2)

    pos3 = ax3.imshow(SumMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig3.colorbar(pos3, ax=ax3, shrink = 0.8)
    ax3.set_title('Sum Events Matrix')
    ax3.set_xlabel('pixels')
    ax3.set_ylabel('pixels')
    xyMax = np.where(SumMatrix >= MaxSumMatrix)
    ax3.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax3)


    pos4 = ax4.imshow(AverageMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig4.colorbar(pos4, ax=ax4, shrink = 0.8)
    ax4.set_title('Average Events Matrix')
    ax4.set_xlabel('pixels')
    ax4.set_ylabel('pixels')
    xyMax = np.where(AverageMatrix >= MaxAvgMatrix)
    ax4.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    xyMin = np.where(AverageMatrix <= MinAvgMatrix)
    ax4.annotate('Min: ('+ str(xyMin[1])+',' + str(xyMin[0])+')', xy=(xyMin[1], xyMin[0]), xytext=(xyMin[1]-75, xyMin[0]-75), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax4)

    plt.show()
    




#Function to display the number of events histogram of the image.
#Could be the positive or negative histogram
def displayHistoNumEvents(arrayEvents):
    print('Function displayHistogramPositiveE. Display an array (positive or negative events) as histogram.')
    #Histogram
    #Variable to define the bins of the histogram
    binWidth = 10
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (max(arrayEvents[:,-1]) + binWidth), binWidth)
    

    fig, ax = plt.subplots()
    ax.hist(histoPositiveEvents[:,2], bins=xbins, edgecolor = 'orange')

    #plot the xdata locations on the x axis:
    ax.plot(arrayEvents[:,2], np.zeros_like(arrayEvents[:,2]), 'd', label = 'Data Points')

    plt.title("Histogram of Image's Number of Events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Number of events')
    ax.set_ylabel('Number of pixels')
    ax.legend()
    displayExtraInfo(ax)
    ax.annotate('Events Bin Width: ' + str(binWidth), xy = (0, -46), xycoords = 'axes points', fontsize = 8)
    plt.show()

#Function to make an array with the pixels and the number of events
#The idea is to replace the function eventsPerPixel() and the two variables histoPositiveEvents, histoNegativeEvents to have just one
#varibale with both positive and negative events
def makePixelsHistogram(xCoord, yCoord, polarity):

    global pixelsEvents
    flag = True
    if( xCoord in pixelsEvents[:,0] ):
        if ( yCoord in pixelsEvents[:,1] ):
##            for u in range(len(pixelsEvents[:,0])):
##                if( (pixelsEvents[u,0] == xCoord) & (pixelsEvents[u,1] == yCoord) ):
##                    
##                    if(polarity == 1):
##                        pixelsEvents[u,2] += 1
##                        pixelsEvents[u,4] += 1
##                        flag = False
##                        break
##                    
##                    if(polarity == 0):
##                        pixelsEvents[u,3] += 1
##                        pixelsEvents[u,4] += 1
##                        flag = False
##                        break
##            
##            if(flag):
##                if(polarity == 1):
##                    pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 1, 0, 1])))
##                if(polarity == 0):
##                    pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 0, 1, 1])))
            index = np.where( (pixelsEvents[:,0] == xCoord) & (pixelsEvents[:,1] == yCoord) )
            if( len( pixelsEvents[index] ) == 1 ):
                if(polarity == 1):
                    pixelsEvents[index,2] += 1
                    pixelsEvents[index,4] += 1
                    
                if(polarity == 0):
                    pixelsEvents[index,3] += 1
                    pixelsEvents[index,4] += 1
            else:
                if(polarity == 1):
                    pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 1, 0, 1])))
                if(polarity == 0):
                    pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 0, 1, 1])))
                        
                
        else:
            if(polarity == 1):
                pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 1, 0, 1])))
            if(polarity == 0):
                pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 0, 1, 1])))
                
    else:
        if(polarity == 1):
            pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 1, 0, 1])))
        if(polarity == 0):
            pixelsEvents = np.vstack((pixelsEvents, np.array([xCoord, yCoord, 0, 1, 1])))

    
#Function to display desired pixels
def displayPixels(array):

    m = np.zeros((numPixelsY + 1, numPixelsX + 1))


    for i in range( len( array[:,0] ) ):
        fillMatrix(m, array[i, 0], array[i, 1], array[i, 4])

    Capella = False
    Jupiter = False
    Betelgeuse = False
    Procyon = False
    Mars = False
    Pollux = False
    s = np.where( (array[:, 0] == 285) & (array[:, 1] == 60) )
    if( len(array[s]) == 1 ):
        Capella = True


    s = np.where( (array[:, 0] == 530) & (array[:, 1] == 91) )
    if( len(array[s]) == 1 ):
        Jupiter = True


    s = np.where( (array[:, 0] == 508) & (array[:, 1] == 266) )
    if( len(array[s]) == 1 ):
        Betelgeuse = True

    s = np.where( (array[:, 0] == 392) & (array[:, 1] == 439) )
    if( len(array[s]) == 1 ):
        Procyon = True

    s = np.where( (array[:, 0] == 221) & (array[:, 1] == 412) )
    if( len(array[s]) == 1 ):
        Mars = True

    s = np.where((array[:, 0] == 259) & (array[:, 1] == 314))
    if (len(array[s]) == 1):
        Pollux = True


    fig1, ax1 = plt.subplots()
   
    pos1 = ax1.imshow(m, cmap = 'cividis_r', interpolation = 'none')
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)#Colorbar
    ax1.set_title('Pixels After Filtering Process')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')

    annotatePixels(array, ax1)
    

    if( Capella ):
        ax1.annotate('Capella', xy=(285, 60), xytext=(285+50, 60+50), arrowprops=dict(facecolor='black', shrink=0.005))

    if ( Jupiter ):
        ax1.annotate('Jupiter', xy=(530, 91), xytext=(530+50, 91+50), arrowprops=dict(facecolor='black', shrink=0.005))

    if( Betelgeuse ):
        ax1.annotate('Betelgeuse', xy=(508, 266), xytext=(508+50, 266+50), arrowprops=dict(facecolor='black', shrink=0.0005))


    if( Procyon ):
        ax1.annotate('Procyon', xy=(392, 439), xytext=(392+50, 439+50), arrowprops=dict(facecolor='black', shrink=0.005))

    if( Mars ):
        ax1.annotate('Mars', xy=(221, 412), xytext=(221+50, 412+50), arrowprops=dict(facecolor='black', shrink=0.005))

    if (Pollux):
        ax1.annotate('Pollux', xy=(259, 314), xytext=(259 + 50, 314 + 50),
                     arrowprops=dict(facecolor='black', shrink=0.005))

    displayExtraInfo(ax1)


    #saveImage()

    plt.show()

#Function to put a value in an specific pixel in a matrix 
def fillMatrix (m, xCoord, yCoord, value):#m: matrix to fill. xCoord: coordinate x pixel. yCoord: coordinate y pixel, value: value to put
    m[yCoord, xCoord] = value





#Function to filter an array
def filterArray(array, value, eventType, condition):
#Parameter array: Array to filter
#Parameter value: Value use to filter
#Parameter eventType : Choose 1 to filter positive events, 2 to filter negative events, 3 to filter total events
#Parameter condition: Choose 1 to filter greather than (>) value, choose 2 to filter less than (<) value

    if(condition == 1):
        mask = array[:, eventType+1] > value
        return array[mask]
        
    if (condition == 2):
        mask = array[:, eventType+1] < value
        return array[mask]

#Function to display histogram with parameters as time and zone of the image
def displayZoneHistogram(binwidth, timeStop, xCoord, yCoord, sizeZone, title):
#Parameter binwidth : Define the bin width for the histogram
#Parameter timeStop : Time limit to display the histogram
#Parameter xCoord : x coordinate of the top leftmost pixel of the selected zone
#Parameter yCoord : y coordinate of the top leftmost pixel of the selected zone
#Parameter sizeZone : Size in pixels of one side of the selected zone. eg 30x30 zone, sizeZone = 30
#Parameter title : Beginning of the figure title. eg if title is = 'star', the figure title will be "star Histogram Events" 
    print('Function displayZoneHistogram')
    #Variable to define the bins of the histogram
    binWidth = binwidth
    #Filter the events by time
    mask = events[:, -1] <= timeStop
    selectedEvents = events[mask]
    #Filter events by zone of the image
    mask = ( selectedEvents[:, 0] >= xCoord ) & (selectedEvents[:, 0] <= (xCoord + (sizeZone - 1)) ) & ( selectedEvents[:,1] >= yCoord ) & (selectedEvents[:,1] <= (yCoord + (sizeZone - 1)) )
    selectedEvents = selectedEvents[mask]

    #Variable to know the time of the las event
    timeLastEvent = selectedEvents[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)

    fig, ax = plt.subplots()
    ax.hist(selectedEvents[:,-1], bins=xbins, edgecolor = 'black')

    #plot the xdata locations on the x axis:
    ax.plot(selectedEvents[:,-1], 0*selectedEvents[:,2], 'd', label = 'Data Points')

    plt.title(title + " Histogram Events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    displayExtraInfo(ax)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -41), xycoords = 'axes points', fontsize = 8)
    ax.annotate('Size of the zone [px]: ' + str(sizeZone) + 'x' + str(sizeZone), xy = (0, -49), xycoords = 'axes points', fontsize = 8)
    ax.annotate('First pixel of the zone: (' + str(xCoord) + ', ' + str(yCoord) + ')', xy = (0, -57), xycoords = 'axes points', fontsize = 8)
    ax.legend()
    #plt.show()


#Function to display bihistogram with parameters as time and zone of the image: histogram of positive and negative events
def displayZoneBihistogram(binwidth, timeStop, xCoord, yCoord, sizeZone, title):
#Parameter binwidth : Define the bin width for the histogram
#Parameter timeStop : Time limit to display the histogram
#Parameter xCoord : x coordinate of the top leftmost pixel of the selected zone
#Parameter yCoord : y coordinate of the top leftmost pixel of the selected zone
#Parameter sizeZone : Size in pixels of one side of the selected zone. eg 30x30 zone, sizeZone = 30
#Parameter title : Beginning of the figure title. eg if title is = 'star', the figure title will be "star Bi-Histogram Events"
    print('Function displayZoneBihistogram')
    #Histogram
    #Variable to define the bins of the histogram
    binWidth = binwidth
    #Filter the events by time
    mask = events[:, -1] <= timeStop
    selectedEvents = events[mask]
    #Filter events by zone of the image
    mask = ( selectedEvents[:, 0] >= xCoord ) & (selectedEvents[:, 0] <= xCoord + sizeZone ) & ( selectedEvents[:,1] >= yCoord ) & (selectedEvents[:,1] <= yCoord + sizeZone )
    selectedEvents = selectedEvents[mask]

    #Variable to know the time of the las event
    timeLastEvent = selectedEvents[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)
    
    
    #Mask for positive events
    positiveMask = selectedEvents[:,2] == 1
    #Mask for negative events
    negativeMask = selectedEvents[:,2] == 0

    #BiHistogram - Histogram for positive and negative events
    fig, ax = plt.subplots()
    #Plot the histogram for positive events
    ax.hist(selectedEvents[:,-1][positiveMask], bins=xbins, edgecolor = 'black', label = 'Positive Events')
    #Plot the histogram for negative events
    ax.hist(selectedEvents[:,-1][negativeMask], weights = -np.ones_like(selectedEvents[:,-1][negativeMask]), bins=xbins, edgecolor = 'black', label = 'Negative Events')

    #Plot the data (positive and negative) along the x axis
    #plot the xdata locations on the x axis:
    #ax.plot(selectedEvents[:,-1][positiveMask], 0*selectedEvents[:,-1][positiveMask], '+', c = 'g', label = 'Positive Data Points')
    #ax.plot(selectedEvents[:,-1][negativeMask], 0*selectedEvents[:,-1][negativeMask], 'o', c = 'k', label = 'Negative Data Points')

    plt.title(title + " Bi-Histogram Events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    ax.legend()
    #Extra info image
    displayExtraInfo(ax)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -41), xycoords = 'axes points', fontsize = 8)
    ax.annotate('Size of the zone [px]: ' + str(sizeZone) + 'x' + str(sizeZone), xy = (0, -49), xycoords = 'axes points', fontsize = 8)
    ax.annotate('First pixel of the zone: (' + str(xCoord) + ', ' + str(yCoord) + ')', xy = (0, -57), xycoords = 'axes points', fontsize = 8)
    #plt.show()


#Function to filter pixels by direct neighbors with a x number of events.
def directNeighbors(array, numMinEvents, numMinNeighbors, neighbors):
#The function search in every pixel in the input array, looking for at least one direct neighbor (up, down, left or right)
#with at least the numMinEVents in any neighbor
#Parameter array : The array wih the pixels to filter
#Parameter numMinEVents : The minimum number of events in the neighbors pixel.
#Parameter numMinNeighbors : The minimum number of neighbors per pixel.
#Parameter neighbors : The number of neighbors to look for. If neighbors=4, the function will look for neighbors above,
#to the right, below and to the left. If neighbors=8, the function will also look for the neighbors in the diagonals.

    outputArray = np.zeros([1, 5], dtype = int)
    for i in range(len(array[:,0])) :
        numNeighbors = 0

        #Direct neighbors
        #Neighbor above
        a = np.where(((array[:, 0]) == (array[i, 0])) & ((array[:, 1]) == ((array[i, 1]) -1 ))) #Find a pixel in the coordinates above
        if( len(array[a]) == 1 ):#There is a neighbor above
            if( array[a, 4] >= numMinEvents ):#There is more or equal numMinEvents
                numNeighbors += 1 #Neighbor number 1

        #Neighbor right
        a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == (array[i, 1]))) #Find a pixel in the coordinates to the right
        if (len(array[a]) == 1):  # There is a neighbor to the right
            if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 2

        # Neighbor below
        a = np.where( (  (array[:, 0])  ==  (array[i, 0])  ) & (  (array[:, 1])  ==  ((array[i, 1]) + 1)  ) ) # Find a pixel in the coordinates below
        if (len(array[a]) == 1):  # There is a neighbor below
            if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 3

        # Neighbor to the left
        a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == (array[i, 1])))  # Find a pixel in the coordinates below
        if (len(array[a]) == 1):  # There is a neighbor to the left
            if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 4

        if ( neighbors == 8 ): #Indirect neighbors

            # Neighbor in the upper right side
            a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ( (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper right side
            if (len(array[a]) == 1):  # There is a neighbor in the upper right side
                if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 1

            # Neighbor in the lower right side
            a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower right side
            if (len(array[a]) == 1):  # There is a neighbor to the lower right side
                if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 2

            # Neighbor in the lower left side
            a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower left side
            if (len(array[a]) == 1):  # There is a neighbor to the lower left side
                if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 3

            # Neighbor in the upper left side
            a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper left side
            if (len(array[a]) == 1):  # There is a neighbor to the upper left side
                if (array[a, 4] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 4



        #Add the pixel if it has the minimum number of neighbors
        if( numNeighbors >= numMinNeighbors ):
            outputArray = np.vstack((outputArray, array[i]))

    outputArray = np.delete(outputArray, 0, 0)
    return outputArray


# Function to identify if two or more pixels who are direct neighbors, belong to a single star.
def isStar(array):
    # The function search in every pixel in the input array, looking for directs neighbors (up, down, left or right)
    # and identify the pixel with the highest number of events as the star
    # Parameter array : The array wih the pixels to filter

    outputArray = np.zeros([1, 5], dtype=int)
    for i in range(len(array[:, 0])):
        star = True
        # Neighbor above
        a = np.where(((array[:, 0]) == (array[i, 0])) & (
                    (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates above
        if (len(array[a]) == 1):  # There is a neighbor above
            if (array[a, 4] > array[i, 4]):  # The neighbor above has more events
                star = False  # The i pixel is not the star

        # Neighbor right
        a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & (
                    (array[:, 1]) == (array[i, 1])))  # Find a pixel in the coordinates to the right
        if (len(array[a]) == 1):  # There is a neighbor to the right
            if (array[a, 4] > array[i, 4]):  # The neighbor to the right has more events
                star = False  # The i pixel is not the star

        # Neighbor below
        a = np.where(((array[:, 0]) == (array[i, 0])) & (
                    (array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates below
        if (len(array[a]) == 1):  # There is a neighbor below
            if (array[a, 4] > array[i, 4]):  # The neighbor below has more events
                star = False  # The i pixel is not the star

        # Neighbor to the left
        a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & (
                    (array[:, 1]) == (array[i, 1])))  # Find a pixel in the coordinates to the left
        if (len(array[a]) == 1):  # There is a neighbor to the left
            if (array[a, 4] > array[i, 4]):  # The neighbor to the left has more events
                star = False  # The i pixel is not the star

        # Neighbor in the upper right side
        a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & (
                    (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper right side
        if (len(array[a]) == 1):  # There is a neighbor in the upper right side
            if (array[a, 4] > array[i, 4]):  # The neighbor in the upper right side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the lower right side
        a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & (
                    (array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower right side
        if (len(array[a]) == 1):  # There is a neighbor in the lower right side
            if (array[a, 4] > array[i, 4]):  # The neighbor in the lower right side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the lower left side
        a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & (
                    (array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower left side
        if (len(array[a]) == 1):  # There is a neighbor in the lower left side
            if (array[a, 4] > array[i, 4]):  # The neighbor in the lower left side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the upper left side
        a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & (
                    (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper left side
        if (len(array[a]) == 1):  # There is a neighbor in the upper left side
            if (array[a, 4] > array[i, 4]):  # The neighbor in the upper left side has more events
                star = False  # The i pixel is not the star

        if (star):
            outputArray = np.vstack((outputArray, array[i]))  # Add the pixel to the output array

    outputArray = np.delete(outputArray, 0, 0)  # Delete the first row of the array because it's 0 0 0 0 0
    return outputArray


#Function to annotate - make more visible some pixels
def annotatePixels(array, ax):
    #The function annotates all the pixels inside the array

    for i in range( len( array[:,0] ) ) :
        ax.annotate(str(i), xy=(array[i,0], array[i,1]), xytext=(array[i,0]+15, array[i,1]+15), arrowprops=dict(facecolor='black', shrink=0.005))



#################################################################################################################################################
#Main Program
#################################################################################################################################################

#Import the data file
# ___________ FOR METEOR 00:29:40 _______________________
file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor_002940.csv' # Modify according to the file path
#file_path = '/users/danidelr86/Téléchargements/ETIS_stars/data_files/meteor.csv' # Modify according to the file path
#file_path = 'D:/Documentos pc Acer/Descargas pc Acer/ETIS/dataFiles/recording_2024-12-13_01-01-22_4min03-to-15min.csv' # Modify according to the file path
#file_path = 'D:/Documentos pc Acer/Descargas pc Acer/ETIS/dataFiles/recording_2024-12-13_01-01-22_24min17-to-37min32.csv'

#Message text
##dataName = file_path[55:-4]
##actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
##fileImgName = dataName + '_' + actualDataTime


#start = time.time()#To count the time to read the file. Delete this
with open(file_path, 'r') as csv_file:#Read the file
    reader = csv.reader(csv_file)
    events = np.array(list(reader), dtype=int)#Originally, dtype=float, I changed to int because there was an error trying 
    #end = time.time()#To count the time to read the file. Delete this
    #print(end - start)#To count the time to read the file. Delete this
    events[:, -1] -= min(events[:, -1])  # Start all sequences at 0.

#Size of the image/matrix
numPixelsX = max(events[ :, 0])
numPixelsY = max(events[ :, 1])
print('The number of pixels in x is', numPixelsX + 1)
print('The number of pixels in y is', numPixelsY + 1)


#Matrix for events
PositiveEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))
NegativeEventsMatrix = np.zeros((numPixelsY + 1, numPixelsX + 1))

  
print('Tha amount of event data is:',len(events))



#Array to count the events per pixel. nx5 = [xCoord, yCoord, positiveEvents, negativeEvents, totalEvents]
#The idea is to replace the function eventsPerPixel() and the two variables histoPositiveEvents, histoNegativeEvents to have just one
#varibale with both positive and negative events
pixelsEvents = np.zeros([1, 5], dtype = int)





#Loop through events (data) array to fill the event matrix and arrays
for i in range(len(events)):
    
    makePixelsHistogram(events[i,0], events[i,1], events[i,2])

    if (events[i,2] == 1):
        CountingEventsPerPixel(PositiveEventsMatrix, events[i,0], events[i,1])

    elif (events[i,2] == 0):
        CountingEventsPerPixel(NegativeEventsMatrix, events[i,0], events[i,1])




#Delete the first row of pixelsEvents because is 0,0,0,0,0
#The idea is to replace the function eventsPerPixel() and the two variables histoPositiveEvents, histoNegativeEvents to have just one
#varibale with both positive and negative events
pixelsEvents = np.delete(pixelsEvents, 0, 0)

#display4Matrix()

# # #Test to filter just the stars and find out if the method works.
remainPixels = directNeighbors(pixelsEvents, 1, 3, 8)
remainPixels = filterArray(remainPixels, 5, 3, 1)
mean = np.ceil(np.mean(remainPixels[:, -1]))
print('The average number of events in the array is : ', mean)
remainPixels = filterArray(remainPixels, mean, 3, 1)
remainPixels = isStar(remainPixels)
print('There are ', len(remainPixels), 'possible stars')
print('The possible stars are: \n', remainPixels)
displayPixels(remainPixels)









# print('The pixels after filtering are :', len(remainPixels))
# print('The minimum number of events in the array is : ', min(remainPixels[:,-1]))
# print('The maximum number of events in the array is : ', max(remainPixels[:,-1]))
















#Verify the number of evets in a pixel
##n = np.where( (pixelsEvents[:,0] == 285) & (pixelsEvents[:,1] == 60) )
##print('pixelCapella: ', pixelsEvents[n])
##n = np.where( (pixelsEvents[:,0] == 530) & (pixelsEvents[:,1] == 91) )
##print('pixelCapella: ', pixelsEvents[n])
##n = np.where( (pixelsEvents[:,0] == 508) & (pixelsEvents[:,1] == 266) )
##print('pixelCapella: ', pixelsEvents[n])
##n = np.where( (pixelsEvents[:,0] == 392) & (pixelsEvents[:,1] == 439) )
##print('pixelCapella: ', pixelsEvents[n])
##n = np.where( (pixelsEvents[:,0] == 221) & (pixelsEvents[:,1] == 412) )
##print('pixelCapella: ', pixelsEvents[n])


#display4Matrix()
#displayHistoNumEvents(histoNegativeEvents)
#displayPixels()
#displayZoneHistogram(20000, events[-1, -1], 584, 387, 20, 'Background')
#displayZoneBihistogram(20000, events[-1, -1], 584, 387, 20, 'Background')
#plt.show()