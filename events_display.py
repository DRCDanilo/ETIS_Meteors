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
from numpy.ma.core import remainder

#matplotlib.use('Qt5Agg') #To have better Matplotlib figures in Ubuntu


#To display an image
import PIL
from PIL import Image


#import sys
#sys.path.append("/")



#################################################################################################################################################
#Functions
#################################################################################################################################################
def counting_events_per_pixel (matrix, x_coord, y_coord):
#Function to fill a matrix with the number of changes per pixel.
#Parameter matrix : matrix to fill.
#Parameter xCoord : coordinate x pixel.
#Parameter yCoord : coordinate y pixel.

    matrix[y_coord, x_coord] += 1



def save_image (file_path):
#Function to save the image.

    dataName = file_path[55:-4]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_' + actualDataTime + '.pdf'

    save_folder = "/users/danidelr86/Téléchargements/ETIS_stars/images/article_20241213T003019"

    full_path = os.path.join(save_folder, fileImgName)
    
    #plt.savefig(full_path,bbox_inches='tight', pad_inches=0)
    plt.savefig(fileImgName,bbox_inches='tight', dpi=600)


def displayExtraInfo (axe, filePath):
#Function to display additional information in the image
    dataName = filePath[55:]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fileImgName = dataName + '_' + actualDataTime
    axe.annotate('Data file: ' + dataName, xy = (0, -25), xycoords = 'axes points', fontsize = 8)
    axe.annotate('Date: ' + actualDataTime, xy = (0, -33), xycoords = 'axes points', fontsize = 8)


def displayHistogram(array, binWidth, filePath):
#Function to display histogram of the input array.
#Parameter array : The array with the pixels to make the histogram of.
#Parameter binWidth : The parameter to define the bin width of the histogram in microseconds.
#Parameter filePath : Variable with the location of the data file.

    print('Function displayHistogram')
    #Histogram
    #Variable to define the bins of the histogram

    #Variable to know the time of the las event
    timeLastEvent = array[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)

    fig, ax = plt.subplots()
    ax.hist(array[:,-1], bins=xbins, edgecolor = 'orange')

    #plot the xdata locations on the x axis:
    ax.plot(array[:,-1], 0*array[:,2], 'd')

    plt.title("Histogram of events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    displayExtraInfo(ax, filePath)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
    plt.show()
    


def displayBihistogram(array, binWidth, filePath):
#Function to display the bi-histogram of the input array: histogram of the positive and negative events.
#Parameter array : The array with the pixels to make the histogram of.
#Parameter binWidth : The parameter to define the bin width of the histogram in microseconds.
#Parameter filePath : Variable with the location of the data file.

    print('Function displayHistogram')
    #Histogram
    #Variable to know the time of the las event
    timeLastEvent = array[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)

    #BiHistogram - Histogram for positive and negative events
    fig, ax = plt.subplots()
    #Mask for positive events
    positiveMask = array[:,2] == 1
    #Mask for negative events
    negativeMask = array[:,2] == 0

    #Plot the histogram for positive events
    ax.hist(array[:,-1][positiveMask], bins=xbins, edgecolor = 'black', label = 'Positive Events')
    #Plot the histogram for negative events
    ax.hist(array[:,-1][negativeMask], weights = -np.ones_like(array[:,-1][negativeMask]), bins=xbins, edgecolor = 'black', label = 'Negative Events')

    #Plot the data (positive and negative) along the x axis
    #plot the xdata locations on the x axis:
    #ax.plot(events[:,-1][positiveMask], 0*events[:,-1][positiveMask], '+', c = 'w')
    #ax.plot(events[:,-1][negativeMask], 0*events[:,-1][negativeMask], 'o', c = 'k')

    plt.title("Histogram of events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Number of events')
    ax.legend()
    #Extra info image
    displayExtraInfo(ax, filePath)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
    plt.show()


def display_4_matrices(positiveMatrix, negativeMatrix, filePath):
#Function to display the 4 matrices of the data:
#Positive events matrix, negative events matrix, total events matrix, average events Matrix
#The function also saves the 4 images automatically in pdf format.
#Parameter positiveMatrix : The positive events matrix.
#Parameter negativeMatrix : The negative events matrix.
#Parameter filePath : The data file path.
    
    #Total events matrix
    SumMatrix = positiveMatrix + negativeMatrix
    #Average events matrix
    AverageMatrix = positiveMatrix - negativeMatrix

    #Display images
    #Display positive events matrix
    #Variable MaxPosMatrix : Max value of the matrix
    MaxPosMatrix = np.max(positiveMatrix)
    print('The maximun number of events in the positive matrix is: ', np.max(positiveMatrix))
    print('The minimun number of events in the positive matrix is: ', np.min(positiveMatrix, where = positiveMatrix > 0, initial = np.inf))

    #Display negative events matrix
    #Variable MaxNegMatrix : Max value of the matrix
    MaxNegMatrix = np.max(negativeMatrix)
    print('The maximun number of events in the negative matrix is: ', np.max(negativeMatrix))
    print('The minimun number of events in the negative matrix is: ', np.min(negativeMatrix, where = negativeMatrix > 0, initial = np.inf))

    #Display total events matrix
    #Variable MaxSumMatrix : Max value of the matrix
    MaxSumMatrix = np.max(SumMatrix)
    print('The maximun number of events in the sum matrix is: ', np.max(SumMatrix))
    print('The minimun number of events in the sum matrix is: ', np.min(SumMatrix, where = SumMatrix > 0, initial = np.inf))

    #Display average events matrix
    #Variable MaxAvgMatrix : Max value of the matrix
    MaxAvgMatrix = np.max(AverageMatrix)
    #Variable MinAvgMatrix : Min value of the matrix
    MinAvgMatrix = np.min(AverageMatrix)
    print('The maximun number of events in the average matrix is: ', np.max(AverageMatrix))
    print('The minimun number of events in the average matrix is: ', np.min(AverageMatrix))

    #Variable vMaxScale : Used as max value in the color scale next to the images
    vMaxScale = 20

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    pos1 = ax1.imshow(positiveMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
    ax1.set_title('Positve Events Matrix')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')
    xyMax = np.where(positiveMatrix >= MaxPosMatrix)
    ax1.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax1, filePath)

    dataName = filePath[55:-4]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_PosMatrix_' + actualDataTime + '.pdf'
    fig1.savefig(fileImgName,bbox_inches='tight', dpi=600)

    pos2 = ax2.imshow(negativeMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig2.colorbar(pos2, ax=ax2, shrink = 0.8)
    ax2.set_title('Negative Events Matrix')
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')
    xyMax = np.where(negativeMatrix >= MaxNegMatrix)
    ax2.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax2, filePath)

    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_NegMatrix_' + actualDataTime + '.pdf'
    fig2.savefig(fileImgName,bbox_inches='tight', dpi=600)

    pos3 = ax3.imshow(SumMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig3.colorbar(pos3, ax=ax3, shrink = 0.8)
    ax3.set_title('Total Events Matrix')
    ax3.set_xlabel('pixels')
    ax3.set_ylabel('pixels')
    xyMax = np.where(SumMatrix >= MaxSumMatrix)
    ax3.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax3, filePath)
    
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_TotMatrix_' + actualDataTime + '.pdf'
    fig3.savefig(fileImgName,bbox_inches='tight', dpi=600)
    
    pos4 = ax4.imshow(AverageMatrix, cmap = 'cividis_r', interpolation = 'none', vmax = vMaxScale)
    fig4.colorbar(pos4, ax=ax4, shrink = 0.8)
    ax4.set_title('Average Events Matrix')
    ax4.set_xlabel('pixels')
    ax4.set_ylabel('pixels')
    xyMax = np.where(AverageMatrix >= MaxAvgMatrix)
    ax4.annotate('Max: ('+ str(xyMax[1])+',' + str(xyMax[0])+')', xy=(xyMax[1], xyMax[0]), xytext=(xyMax[1]+50, xyMax[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    xyMin = np.where(AverageMatrix <= MinAvgMatrix)
    ax4.annotate('Min: ('+ str(xyMin[1])+',' + str(xyMin[0])+')', xy=(xyMin[1], xyMin[0]), xytext=(xyMin[1]-75, xyMin[0]-75), arrowprops=dict(facecolor='black', shrink=0.025))
    displayExtraInfo(ax4, filePath)

    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_AvgMatrix_' + actualDataTime + '.pdf'
    fig4.savefig(fileImgName,bbox_inches='tight', dpi=600)
    
    plt.show()
    





def displayHistoNumEvents(arrayEvents):
#Function to display the number of events histogram of the image.
#Could be the positive or negative histogram.
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


def makePixelsHistogram(xCoord, yCoord, polarity, array):
#Function to make an array with all the pixels in the input data file, with the total number of events (positives and
#negatives. The output is an array with all the n pixels in the data input, and their total events number for all the
#file duration.
#Parameter xCoord : The coordinate x of the pixel.
#Parameter yCoord : The coordinate y of the pixel.
#Parameter polarity : The polarity of the event.
#Parameter array : The array to store all the pixels and their events.

    if( xCoord in array[:,0] ): #Look for the coordinate x of the pixel in the array
        if ( yCoord in array[:,1] ): #Look for the coordinate y of the pixel in the array

            index = np.where( (array[:,0] == xCoord) & (array[:,1] == yCoord) ) #Get the position on the array
            if( len( array[index] ) == 1 ): #Check the position on the array
                if(polarity == 1):
                    array[index,2] += 1
                    array[index,4] += 1
                    
                if(polarity == 0):
                    array[index,3] += 1
                    array[index,4] += 1
            else:
                if(polarity == 1):
                    array = np.vstack((array, np.array([xCoord, yCoord, 1, 0, 1]))) #Add the pixel to the array
                if(polarity == 0):
                    array = np.vstack((array, np.array([xCoord, yCoord, 0, 1, 1]))) #Add the pixel to the array
        else:
            if(polarity == 1):
                array = np.vstack((array, np.array([xCoord, yCoord, 1, 0, 1]))) #Add the pixel to the array
            if(polarity == 0):
                array = np.vstack((array, np.array([xCoord, yCoord, 0, 1, 1]))) #Add the pixel to the array
    else:
        if(polarity == 1):
            array = np.vstack((array, np.array([xCoord, yCoord, 1, 0, 1]))) #Add the pixel to the array
        if(polarity == 0):
            array = np.vstack((array, np.array([xCoord, yCoord, 0, 1, 1]))) #Add the pixel to the array

    return array

    

def displayPixels(array, matrix, filePath):
#Function to display desired pixels.

    #m = np.zeros((numPixelsY + 1, numPixelsX + 1))


    for i in range( len( array[:,0] ) ):
        fillMatrix(matrix, array[i, 0], array[i, 1], array[i, 4])

    maxValueArray = np.max(array[:, 4])


    # Capella = False
    # Jupiter = False
    # Betelgeuse = False
    # Procyon = False
    # Mars = False
    # Pollux = False
    # s = np.where( (array[:, 0] == 285) & (array[:, 1] == 60) )
    # if( len(array[s]) == 1 ):
    #     Capella = True
    #
    #
    # s = np.where( (array[:, 0] == 530) & (array[:, 1] == 91) )
    # if( len(array[s]) == 1 ):
    #     Jupiter = True
    #
    #
    # s = np.where( (array[:, 0] == 508) & (array[:, 1] == 266) )
    # if( len(array[s]) == 1 ):
    #     Betelgeuse = True
    #
    # s = np.where( (array[:, 0] == 392) & (array[:, 1] == 439) )
    # if( len(array[s]) == 1 ):
    #     Procyon = True
    #
    # s = np.where( (array[:, 0] == 221) & (array[:, 1] == 412) )
    # if( len(array[s]) == 1 ):
    #     Mars = True
    #
    # s = np.where((array[:, 0] == 259) & (array[:, 1] == 314))
    # if (len(array[s]) == 1):
    #     Pollux = True


    fig1, ax1 = plt.subplots()
   
    pos1 = ax1.imshow(matrix, cmap = 'cividis_r', interpolation = 'none', vmax = maxValueArray)
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)#Colorbar
    ax1.set_title('Pixels After Filtering Process')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')

    annotatePixels(array, ax1)
    

    # if( Capella ):
    #     ax1.annotate('Capella', xy=(285, 60), xytext=(285+50, 60+50), arrowprops=dict(facecolor='black', shrink=0.005))
    #
    # if ( Jupiter ):
    #     ax1.annotate('Jupiter', xy=(530, 91), xytext=(530+50, 91+50), arrowprops=dict(facecolor='black', shrink=0.005))
    #
    # if( Betelgeuse ):
    #     ax1.annotate('Betelgeuse', xy=(508, 266), xytext=(508+50, 266+50), arrowprops=dict(facecolor='black', shrink=0.0005))
    #
    #
    # if( Procyon ):
    #     ax1.annotate('Procyon', xy=(392, 439), xytext=(392+50, 439+50), arrowprops=dict(facecolor='black', shrink=0.005))
    #
    # if( Mars ):
    #     ax1.annotate('Mars', xy=(221, 412), xytext=(221+50, 412+50), arrowprops=dict(facecolor='black', shrink=0.005))
    #
    # if (Pollux):
    #     ax1.annotate('Pollux', xy=(259, 314), xytext=(259 + 50, 314 + 50),
    #                  arrowprops=dict(facecolor='black', shrink=0.005))

    displayExtraInfo(ax1, filePath)


    #saveImage()

    plt.show()

def fill_matrix (matrix, x_coord, y_coord, value):
#Function to put a value in a specific pixel in a matrix.
#Parameter matrix : matrix to fill.
#Parameter x_coord : coordinate x pixel.
#Parameter y_coord : coordinate y pixel.

    matrix[y_coord, x_coord] = value

def filterArray(array, value, eventType, condition):
#Function to filter an array.
#Parameter array: Array to filter
#Parameter value: Value use to filter
#Parameter eventType : Choose 1 to filter positive events, 2 to filter negative events, 3 to filter total events. The
#function filter the column number eventType+1, so if I want to filter the events in the column 6 of the array, Python
# indexs start at 0, so the parameter eventType would be 4.
#Parameter condition: Choose 1 to filter greater than (>) value, choose 2 to filter less than (<) value

    if(condition == 1):
        mask = array[:, eventType + 1 ] > value
        return array[mask]
        
    if (condition == 2):
        mask = array[:, eventType + 1 ] < value
        return array[mask]


def displayZoneHistogram(binwidth, timeStop, xCoord, yCoord, sizeZone, title):
#Function to display histogram with parameters as time and zone of the image.
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
    #xbins = np.arange(0, (timeLastEvent + binWidth), binWidth) # This line works ok for the histogram until the time of last event
    xbins = np.arange(0, timeStop, binWidth) # This line is a test to have the histogram until the time stop no matter the presence of events

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



def displayZoneBihistogram(binwidth, timeStop, xCoord, yCoord, sizeZone, title):
#Function to display bihistogram with parameters as time and zone of the image: histogram of positive and negative events.
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
    mask = ( selectedEvents[:, 0] >= xCoord ) & (selectedEvents[:, 0] <= (xCoord + (sizeZone - 1)) ) & ( selectedEvents[:,1] >= yCoord ) & (selectedEvents[:,1] <= (yCoord + (sizeZone - 1)) )
    selectedEvents = selectedEvents[mask]

    #Variable to know the time of the las event
    timeLastEvent = selectedEvents[-1,-1]
    #Build an array with the bins for the histogram
    #xbins = np.arange(0, (timeLastEvent + binWidth), binWidth) # This line works ok for the histogram until the time of last event
    xbins = np.arange(0, timeStop, binWidth) # This line is a test to have the histogram until the time stop no matter the presence of events
    
    
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
    ax.plot(selectedEvents[:,-1][positiveMask], 0*selectedEvents[:,-1][positiveMask], '+', c = 'g', label = 'Positive Data Points')
    ax.plot(selectedEvents[:,-1][negativeMask], 0*selectedEvents[:,-1][negativeMask], 'o', c = 'k', label = 'Negative Data Points')

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



def directNeighbors(array, numMinEvents, numMinNeighbors, neighbors, numColumn):
#Function to filter pixels by direct neighbors with a x number of events.
#The function search in every pixel in the input array, looking for at least one direct neighbor (up, down, left or right)
#with at least the numMinEVents in any neighbor
#Parameter array : The array wih the pixels to filter
#Parameter numMinEVents : The minimum number of events in the neighbors pixel.
#Parameter numMinNeighbors : The minimum number of neighbors per pixel.
#Parameter neighbors : The number of neighbors to look for. If neighbors=4, the function will look for neighbors above,
#to the right, below and to the left. If neighbors=8, the function will also look for the neighbors in the diagonals.
#Parameter numColumn : The number of column in the array to look for the events. i.e. 4=total events, 5=events x timeUnit

    #outputArray = np.zeros([1, 5], dtype = int) #Output array : xCoord, yCoord, positiEVents, negatiEvents, totalEvents, events x timeUnit
    outputArray = np.zeros([1, array.shape[1]], dtype = int) #Output array : xCoord, yCoord, positiEVents, negatiEvents, totalEvents, events x timeUnit
    for i in range(len(array[:,0])) :
        numNeighbors = 0

        #Direct neighbors
        #Neighbor above
        a = np.where(((array[:, 0]) == (array[i, 0])) & ((array[:, 1]) == ((array[i, 1]) -1 ))) #Find a pixel in the coordinates above
        if( len(array[a]) == 1 ):#There is a neighbor above
            if( array[a, numColumn] >= numMinEvents ):#There is more or equal numMinEvents
                numNeighbors += 1 #Neighbor number 1

        #Neighbor right
        a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == (array[i, 1]))) #Find a pixel in the coordinates to the right
        if (len(array[a]) == 1):  # There is a neighbor to the right
            if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 2

        # Neighbor below
        a = np.where( (  (array[:, 0])  ==  (array[i, 0])  ) & (  (array[:, 1])  ==  ((array[i, 1]) + 1)  ) ) # Find a pixel in the coordinates below
        if (len(array[a]) == 1):  # There is a neighbor below
            if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 3

        # Neighbor to the left
        a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == (array[i, 1])))  # Find a pixel in the coordinates below
        if (len(array[a]) == 1):  # There is a neighbor to the left
            if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                numNeighbors += 1 # Neighbor number 4

        if ( neighbors == 8 ): #Indirect neighbors

            # Neighbor in the upper right side
            a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ( (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper right side
            if (len(array[a]) == 1):  # There is a neighbor in the upper right side
                if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 1

            # Neighbor in the lower right side
            a = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower right side
            if (len(array[a]) == 1):  # There is a neighbor to the lower right side
                if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 2

            # Neighbor in the lower left side
            a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a pixel in the coordinates to the lower left side
            if (len(array[a]) == 1):  # There is a neighbor to the lower left side
                if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 3

            # Neighbor in the upper left side
            a = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) - 1)))  # Find a pixel in the coordinates to the upper left side
            if (len(array[a]) == 1):  # There is a neighbor to the upper left side
                if (array[a, numColumn] >= numMinEvents):  # There is more or equal numMinEvents
                    numNeighbors += 1  # Indirect neighbor number 4



        #Add the pixel if it has the minimum number of neighbors
        if( numNeighbors >= numMinNeighbors ):
            outputArray = np.vstack((outputArray, array[i]))

    outputArray = np.delete(outputArray, 0, 0)
    return outputArray



def isStar(array):
#Function to identify if two or more pixels who are direct neighbors, belong to a single star.
#The function search in every pixel in the input array, looking for directs neighbors (up, down, left or right)
#and identify the pixel with the highest number of events as the star
#Parameter array : The array wih the pixels to filter

    outputArray = np.zeros([1, array.shape[1]], dtype=int)
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



def annotatePixels(array, ax):
#Function to annotate - make more visible some pixels
#The function annotates all the pixels inside the array

    for i in range( len( array[:,0] ) ) :
        ax.annotate(str(i), xy=(array[i,0]+3, array[i,1]), xytext=(array[i,0]+12, array[i,1]), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))




def continuity(array, e):
#Function to analyze along the time an array and determine if here is continuity of events.
#The function look for continuity in the array, that means look for a number of events in the array. The number of events is determined by the parameter e.
#Parameter array : The array to analyze
#Parameter e : The minimum number of events in the array to have continuity

    if( len(array) >= e ):
        return True
    return None



def continuousStar(array, interval, timeStop, level):
#Function to identify a star looking for the continuity of the events in a time interval.
#Parameter array : The array where the possible stars are.
#Parameter interval : The time interval to evaluate the continuity of the events.
#Parameter timeStop : The time limit to look for the continuity of the events.
#Parameter level : The minimum number of evidences to consider a pixel a star.

    windows = np.ceil(timeStop / interval) # Number of time windows to make the filtering
    outputArray = np.zeros([1, 5], dtype = int) # Output array

    for i in range( len (array[:, 0]) ): # Go through the input array
        index = np.where( ( events[:, 0] == array[i, 0] ) & ( events[:, 1] == array[i, 1] ) ) # Find pixel events in all the events data
        onePixelIndexs = np.array(index) # Convert to array the tuple with the indices of the pixel events
        onePixelEvents = np.take(events, onePixelIndexs, axis=0) # Extract only the events of the pixel from all events data
        onePixelEvents = np.reshape(onePixelEvents,(onePixelEvents.shape[1],onePixelEvents.shape[2])) # Build the array (nx4) of the pixel events
        onePixelEvents[: , -1] -= min(onePixelEvents[:,  -1]) # Start the pixel events from t = 0

        proofs = 0
        for j in range(int(windows)): # Go through the windows or time intervals
            mask = ( ( (onePixelEvents[:, -1]) >= (j * interval) ) & ( (onePixelEvents[:, -1]) < ( (j + 1) * interval ) ) ) # Select only the events in the time interval
            pixelsSliced = onePixelEvents[mask] # Array with the events in the time interval
            # Function to analyze the pixels in the time interval
            mark = continuity(pixelsSliced, 2)
            if mark: # If the array in the time interval corresponds to a star
                proofs = proofs + 1
        if proofs >= level:
            outputArray = np.vstack((outputArray, array[i]))

    outputArray = np.delete(outputArray, 0, 0)
    return outputArray



def eventsByTime(numTotalEvents, totalTimeData, unitOfTime):
#Function to calculate the number of events per second or per a determined amount of time, of one pixel.
#Parameter numTotalEvents : The total number of events of the pixel.
#Parameter totalTimeData : The total duration time of the data.
#Parameter unitOfTime : The time unit for the reference in microseconds.

    eventsPerTime = ( unitOfTime * numTotalEvents ) / ( totalTimeData )
    return eventsPerTime


def addEventsByTime(array, totalTimeData, unitOfTime):
#Function to add the number of events per second or per a determined amount of time, to an array of pixels.
#Parameter array : The array with pixels to add their number of events per amount of time.
#Parameter totalTimeData : The total duration time of the data.
#Parameter unitOfTime : The time unit for the reference.

    for i in range( len( array ) ):
        numToAdd = eventsByTime( array[i, 4], totalTimeData, unitOfTime)
        array[i, 5] = numToAdd
    return array


def star_nan(array, matrix):
#Function to fill with NaN a star pixel and its 8 neighbors pixels.
#Parameter array : The star to fill with NaN, in form of an array with the coordinates in the form = [x,y,...].
#Parameter matrix : The matrix to fill with the NaN terms in.

    fill_matrix(matrix, array[0], array[1], np.nan) #Star pixel
    fill_matrix(matrix, ( array[0] + 1 ), array[1], np.nan)  #Right neighbor pixel
    fill_matrix(matrix, ( array[0] + 1 ), ( array[1] + 1 ), np.nan)  #Right-down neighbor pixel
    fill_matrix(matrix, array[0], ( array[1] + 1 ), np.nan)  #Down neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), ( array[1] + 1 ), np.nan)  # Left-down neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), array[1], np.nan)  # Left neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), ( array[1] - 1 ), np.nan)  # Left-up neighbor pixel
    fill_matrix(matrix, array[0], ( array[1] - 1 ), np.nan)  # Up neighbor pixel
    fill_matrix(matrix, ( array[0] + 1 ), ( array[1] - 1 ), np.nan)  # Right-up neighbor pixel


def rectangle_nan(x1, x2, y1, y2, matrix):
#Function to fill a rectangle (the meteor space) with NaN terms.
#Parameter x1 : The x coordinate where the rectangle starts.
#Parameter x2 : The x coordinate where the rectangle ends.
#Parameter y1 : The y coordinate where the rectangle starts.
#Parameter y2 : The y coordinate where the rectangle ends.
#Parameter matrix : The matrix to form the mask in.

    for j in range(y2-y1):
        for i in range(x2-x1):
            fill_matrix(matrix, i+x1, j+y1, np.nan)


def neighborsOnes(array, matrix):
#Function to fill with ones (1) terms the 8 neighbors pixels of a star. The star pixel will not be filled.
#Parameter array : The star to fill the neighbors with ones (1) terms, in form of an array with the coordinates in the form = [x,y,p,t,.].
#Parameter matrix : The matrix to form the mask in.

    fillMatrix(matrix, ( array[0] + 1 ), array[1], 1)  #Right neighbor pixel
    fillMatrix(matrix, ( array[0] + 1 ), ( array[1] + 1 ), 1)  #Right-down neighbor pixel
    fillMatrix(matrix, array[0], ( array[1] + 1 ), 1)  #Down neighbor pixel
    fillMatrix(matrix, ( array[0] - 1 ), ( array[1] + 1 ), 1)  # Left-down neighbor pixel
    fillMatrix(matrix, ( array[0] - 1 ), array[1], 1)  # Left neighbor pixel
    fillMatrix(matrix, ( array[0] - 1 ), ( array[1] - 1 ), 1)  # Left-up neighbor pixel
    fillMatrix(matrix, array[0], ( array[1] - 1 ), 1)  # Up neighbor pixel
    fillMatrix(matrix, ( array[0] + 1 ), ( array[1] - 1 ), 1)  # Right-up neighbor pixel


def nan_mask(stars_array, input_matrix):
#Function to create a matrix with NaN terms in the star pixels, neighbor star pixels, and a rectangle where the meteor is, using other functions.
#Parameter stars_array : Array with all the stars. Each element in the array should have the form [x,y,.,.,] where x and y are the coordinates of the star.
#Parameter input_matrix : The input matrix where the mask will be.

    for r in range( len( stars_array ) ):
        star_nan( stars_array[r], input_matrix )

    rectangle_nan(73, 93, 456, 480, input_matrix) # NaN square for the meteor of first file meteor.csv
    #rectangle_nan(223, 231, 306, 327, input_matrix) # NaN square for the meteor of second file meteor_003019_long.csv

    return input_matrix

def one_mask(stars_array, input_matrix):
#Function to create a matrix with 1 terms in the neighbor star pixels, using another function.
#Parameter stars_array : Array with all the stars. Each element in the array should have the form [x,y,.,.,] where x and y are the coordinates of the star.
#Parameter input_matrix : The input matrix where the mask will be.

    for r in range( len( stars_array ) ):
        neighbor_one(stars_array[r], input_matrix)

    return input_matrix


def get_parameters(array):
#Function to calculate the average, the median, the minimum and the maximum of an array of pixels.
#Parameter array : The array with the pixels to calculate the parameters for.

    lengthData = len(array)

    maskZero = array == 0
    arrayZero = array[maskZero]

    maskNoZero = array != 0
    arrayNoZero = array[maskNoZero]

    arraySum = np.sum(array, dtype=int)

    totalMean = np.mean(array)

    totalMedian = np.median(array)

    totalMin = np.min(array)

    totalMax = np.max(array)

    print("The length of the pixels' array is: ", lengthData)
    print('The total number of pixels with zero events is : ', len(arrayZero))
    print('The total number of pixels with NON-ZERO events is : ', len(arrayNoZero))
    print("The total sum of pixels' events is : ", arraySum)
    print("The mean of pixels' events is : ", totalMean)
    print("The median of pixels' events is : ", totalMedian)
    print("The number minimum of events in the array is : ", totalMin)
    print("The number maximum of events in the array is : ", totalMax)


# def isMeteor(array):
#Function to identify a meteor trajectory form an array.
#Parameter array : The array with the pixels to identify the trajectory of.


# Function to identify a pixel alone without neighbors pixels and delete it form an array.
def noPixelAlone(array):
# Parameter array : The array with the pixels to identify the pixels alone of.

    outputArray = np.zeros([1, array.shape[1]], dtype=int)
    # for i in range( len( array[:, 0] ) ):
    #     for j in range( 8 ):

def starCoordinatesList(array):
#Function to make a list with the coordinates of the stars and their 8 direct neighbors, who are in the array input of this function.
#Parameter array : The array with the pixels of the stars.

    outputArray = np.zeros([1,2], dtype=int) #Create the output array.

    for i in range( len(array) ): #Go through the input array.
        outputArray = np.vstack((outputArray, np.array([array[i,0], array[i,1]]))) #Add the star coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0], array[i, 1] - 1]))) #Add the neighbor above coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1] - 1]))) #Add the upper right side neighbor coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1]]))) #Add the right neighbor coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1] + 1]))) #Add the lower right side neighbor coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0], array[i, 1] + 1])))  #Add the neighbor below coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1] + 1])))  #Add the lower left side neighbor coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1]])))  # Add the left neighbor coordinate to the output array.
        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1] - 1])))  # Add the upper left side neighbor coordinate to the output array.

    outputArray = np.delete(outputArray, 0, 0) #Delete the first row because it is [0,0].
    return outputArray

def meteorCoordinatesList(x1, x2, y1, y2):
#Function to make a list with the coordinates of the rectangle where the meteor's trajectory is.
#Parameter x1 : The x coordinate where the rectangle starts.
#Parameter x2 : The x coordinate where the rectangle ends.
#Parameter y1 : The y coordinate where the rectangle starts.
#Parameter y2 : The y coordinate where the rectangle ends.

    outputArray = np.zeros([1,2], dtype=int) #Create the output array.

    for j in range((y2-y1)+1):
        for i in range((x2-x1)+1):
            outputArray = np.vstack((outputArray, np.array([i+x1, j+y1]))) #Add the coordinate to the output array.

    outputArray = np.delete(outputArray, 0, 0)  # Delete the first row because it is [0,0].
    return outputArray

def histogram_num_events(array, bin_width, file_path):
#Function to display an histogram of the total events of an input array.
#This function was made for the analysis of the real sky's pixels.
#By default, just the y axis is in log scale. It is necessary to uncomment some lines to have the x axis in log scale too.
#The function also saves the image automatically in pdf format.
#Parameter array : The array with the pixels to make the histogram of.
#Parameter bin_width : The parameter to define the bin width of the histogram in microseconds.
#Parameter file_path : Variable with the location of the data file.

    #Variable to know the max number of events in the array (events of crazy pixel)
    max_event_array = max(array)
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (max_event_array + bin_width), bin_width) #Bins in linear (normal) scale
    #Build an array with the bins for the histogram in log scale
    bins_log = np.logspace(np.log10(array.min()), np.log10(array.max()), 40) #Modify the number 70 to change the scale
    
    #Create the figure
    fig, ax = plt.subplots()
    ax.hist(array[:], bins=xbins, edgecolor = 'orange') #x axis in log scale -> bins = bins_log
    plt.yscale('log') #y axis in log scale
    #plt.xscale('log') #Uncomment to have x axis in log scale
    #plot the xdata locations on the x axis:
    ax.plot(array[:], 0*array[:], 'd')

    plt.title("Histogram of Total Events : Real Sky")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Events')
    ax.set_ylabel('Number of pixels')
    displayExtraInfo(ax, file_path)
    ax.annotate('Events Bin Width: ' + str(bin_width) + ' events', xy = (0, -46), xycoords = 'axes points', fontsize = 8)

    #To save the image
    dataName = file_path[55:-4]
    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    fileImgName = dataName + '_HistoRealSky_' + actualDataTime + '.pdf'
    fig.savefig(fileImgName,bbox_inches='tight', dpi=600)
    
    plt.show()
