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



##def save_image (file_path):
###Function to save the image.
##
##    dataName = file_path[55:-4]
##    actualDataTime = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
##    fileImgName = dataName + '_' + actualDataTime + '.pdf'
##
##    save_folder = "/users/danidelr86/Téléchargements/ETIS_stars/images/article_20241213T003019"
##
##    full_path = os.path.join(save_folder, fileImgName)
##    
##    #plt.savefig(full_path,bbox_inches='tight', pad_inches=0)
##    plt.savefig(fileImgName,bbox_inches='tight', dpi=600)

def display_extra_info (axe, file_path):
#Function to display additional information in the bottom part of the image
#The input data file name, the date and time when the image was generated are displayed
    
    data_name = file_path[55:]
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    axe.annotate('Data file: ' + data_name, xy = (0, -25), xycoords = 'axes points', fontsize = 8)
    axe.annotate('Date: ' + actual_data_time, xy = (0, -33), xycoords = 'axes points', fontsize = 8)

##def displayHistogram(array, binWidth, filePath):
###Function to display histogram of the input array.
###Parameter array : The array with the pixels to make the histogram of.
###Parameter binWidth : The parameter to define the bin width of the histogram in microseconds.
###Parameter filePath : Variable with the location of the data file.
##
##    print('Function displayHistogram')
##    #Histogram
##    #Variable to define the bins of the histogram
##
##    #Variable to know the time of the las event
##    timeLastEvent = array[-1,-1]
##    #Build an array with the bins for the histogram
##    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)
##
##    fig, ax = plt.subplots()
##    ax.hist(array[:,-1], bins=xbins, edgecolor = 'orange')
##
##    #plot the xdata locations on the x axis:
##    ax.plot(array[:,-1], 0*array[:,2], 'd')
##
##    plt.title("Histogram of events")
##    plt.grid(visible = True, color = 'r')
##    ax.set_xlabel('Time [ms]')
##    ax.set_ylabel('Number of events')
##    display_extra_info(ax, filePath)
##    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
##    plt.show()
    
##def displayBihistogram(array, binWidth, filePath):
###Function to display the bi-histogram of the input array: histogram of the positive and negative events.
###Parameter array : The array with the pixels to make the histogram of.
###Parameter binWidth : The parameter to define the bin width of the histogram in microseconds.
###Parameter filePath : Variable with the location of the data file.
##
##    print('Function displayHistogram')
##    #Histogram
##    #Variable to know the time of the las event
##    timeLastEvent = array[-1,-1]
##    #Build an array with the bins for the histogram
##    xbins = np.arange(0, (timeLastEvent + binWidth), binWidth)
##
##    #BiHistogram - Histogram for positive and negative events
##    fig, ax = plt.subplots()
##    #Mask for positive events
##    positiveMask = array[:,2] == 1
##    #Mask for negative events
##    negativeMask = array[:,2] == 0
##
##    #Plot the histogram for positive events
##    ax.hist(array[:,-1][positiveMask], bins=xbins, edgecolor = 'black', label = 'Positive Events')
##    #Plot the histogram for negative events
##    ax.hist(array[:,-1][negativeMask], weights = -np.ones_like(array[:,-1][negativeMask]), bins=xbins, edgecolor = 'black', label = 'Negative Events')
##
##    #Plot the data (positive and negative) along the x axis
##    #plot the xdata locations on the x axis:
##    #ax.plot(events[:,-1][positiveMask], 0*events[:,-1][positiveMask], '+', c = 'w')
##    #ax.plot(events[:,-1][negativeMask], 0*events[:,-1][negativeMask], 'o', c = 'k')
##
##    plt.title("Histogram of events")
##    plt.grid(visible = True, color = 'r')
##    ax.set_xlabel('Time [ms]')
##    ax.set_ylabel('Number of events')
##    ax.legend()
##    #Extra info image
##    display_extra_info(ax, filePath)
##    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -46), xycoords = 'axes points', fontsize = 8)
##    plt.show()

def display_4_matrices(positive_matrix, negative_matrix, file_path):
#Function to display the 4 matrices of the data:
#Positive events matrix, negative events matrix, total events matrix, average events Matrix
#The function also saves the 4 images automatically in pdf format
#Parameter positive_matrix : The positive events matrix
#Parameter negative_matrix : The negative events matrix
#Parameter file_path : The data file path
    
    #Total events matrix
    total_matrix = positive_matrix + negative_matrix
    #Average events matrix
    average_matrix = positive_matrix - negative_matrix

    #Max value of the positive matrix
    max_positive_matrix = np.max(positive_matrix)
    print('The maximum number of events in the positive matrix is: ', np.max(positive_matrix))
    print('The minimum number of events in the positive matrix is: ', np.min(positive_matrix, where = positive_matrix > 0, initial = np.inf))

    #Max value of the negative matrix
    max_negative_matrix = np.max(negative_matrix)
    print('The maximum number of events in the negative matrix is: ', np.max(negative_matrix))
    print('The minimum number of events in the negative matrix is: ', np.min(negative_matrix, where = negative_matrix > 0, initial = np.inf))

    #Max value of the total matrix
    max_total_matrix = np.max(total_matrix)
    print('The maximum number of events in the total matrix is: ', np.max(total_matrix))
    print('The minimum number of events in the total matrix is: ', np.min(total_matrix, where = total_matrix > 0, initial = np.inf))

    #Max value of the average matrix
    max_average_matrix = np.max(average_matrix)
    #Min value of the average matrix
    min_average_matrix = np.min(average_matrix)
    print('The maximum number of events in the average matrix is: ', np.max(average_matrix))
    print('The minimum number of events in the average matrix is: ', np.min(average_matrix))

    #Maximum value in the color scale next to the images
    scale_maximum_value = 20

    #Display images
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    #Display positive events matrix
    pos1 = ax1.imshow(positive_matrix, cmap = 'cividis_r', interpolation = 'none', vmax = scale_maximum_value)
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
    ax1.set_title('Positve Events Matrix')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')
    location_max_pixel = np.where(positive_matrix >= max_positive_matrix)
    ax1.annotate('Max: ('+ str(location_max_pixel[1])+',' + str(location_max_pixel[0])+')', xy=(location_max_pixel[1], location_max_pixel[0]), xytext=(location_max_pixel[1]+50, location_max_pixel[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    display_extra_info(ax1, file_path)

    data_name = file_path[55:-4]
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_PosMatrix_' + actual_data_time + '.pdf'
    fig1.savefig(file_image_name, bbox_inches='tight', dpi=600)

    #Display negative events matrix
    pos2 = ax2.imshow(negative_matrix, cmap = 'cividis_r', interpolation = 'none', vmax = scale_maximum_value)
    fig2.colorbar(pos2, ax=ax2, shrink = 0.8)
    ax2.set_title('Negative Events Matrix')
    ax2.set_xlabel('pixels')
    ax2.set_ylabel('pixels')
    location_max_pixel = np.where(negative_matrix >= max_negative_matrix)
    ax2.annotate('Max: ('+ str(location_max_pixel[1])+',' + str(location_max_pixel[0])+')', xy=(location_max_pixel[1], location_max_pixel[0]), xytext=(location_max_pixel[1]+50, location_max_pixel[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    display_extra_info(ax2, file_path)

    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_NegMatrix_' + actual_data_time + '.pdf'
    fig2.savefig(file_image_name, bbox_inches='tight', dpi=600)

    #Display total events matrix
    pos3 = ax3.imshow(total_matrix, cmap = 'cividis_r', interpolation = 'none', vmax = scale_maximum_value)
    fig3.colorbar(pos3, ax=ax3, shrink = 0.8)
    ax3.set_title('Total Events Matrix')
    ax3.set_xlabel('pixels')
    ax3.set_ylabel('pixels')
    location_max_pixel = np.where(total_matrix >= max_total_matrix)
    ax3.annotate('Max: ('+ str(location_max_pixel[1])+',' + str(location_max_pixel[0])+')', xy=(location_max_pixel[1], location_max_pixel[0]), xytext=(location_max_pixel[1]+50, location_max_pixel[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    display_extra_info(ax3, file_path)
    
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_TotMatrix_' + actual_data_time + '.pdf'
    fig3.savefig(file_image_name, bbox_inches='tight', dpi=600)

    #Display average events matrix
    pos4 = ax4.imshow(average_matrix, cmap = 'cividis_r', interpolation = 'none', vmax = scale_maximum_value)
    fig4.colorbar(pos4, ax=ax4, shrink = 0.8)
    ax4.set_title('Average Events Matrix')
    ax4.set_xlabel('pixels')
    ax4.set_ylabel('pixels')
    location_max_pixel = np.where(average_matrix >= max_average_matrix)
    ax4.annotate('Max: ('+ str(location_max_pixel[1])+',' + str(location_max_pixel[0])+')', xy=(location_max_pixel[1], location_max_pixel[0]), xytext=(location_max_pixel[1]+50, location_max_pixel[0]+50), arrowprops=dict(facecolor='black', shrink=0.025))
    xyMin = np.where(average_matrix <= min_average_matrix)
    ax4.annotate('Min: ('+ str(xyMin[1])+',' + str(xyMin[0])+')', xy=(xyMin[1], xyMin[0]), xytext=(xyMin[1]-75, xyMin[0]-75), arrowprops=dict(facecolor='black', shrink=0.025))
    display_extra_info(ax4, file_path)

    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_AvgMatrix_' + actual_data_time + '.pdf'
    fig4.savefig(file_image_name, bbox_inches='tight', dpi=600)
    
    plt.show()
    
def count_pixel_events(x_coord, y_coord, polarity, array):
#Function to make an array with all the pixels in the input data file, with their total number of events.
#The goal is to have an array that answer the question "How many events does each pixel in the input file have?"
#Parameter x_coord : The x coordinate of the pixel
#Parameter y_coord : The y coordinate of the pixel
#Parameter polarity : The polarity of the event
#Parameter array : The array to store all the pixels and their events

    if( x_coord in array[:,0] ): #Look for the x coordinate of the pixel in the array
        if ( y_coord in array[:,1] ): #Look for the y coordinate of the pixel in the array

            index = np.where( (array[:,0] == x_coord) & (array[:,1] == y_coord) ) #If the pixel is in the array, get the position on the array
            if( len( array[index] ) == 1 ): #Check the position on the array
                if(polarity == 1):
                    array[index,2] += 1 #Add an event to the positive events column of the pixel in the array
                    array[index,4] += 1 #Add an event to the total events column of the pixel in the array
                    
                if(polarity == 0):
                    array[index,3] += 1 #Add an event to the negative events column of the pixel in the array
                    array[index,4] += 1 #Add an event to the total events column of the pixel in the array
            else: #If there is a pixel with the x coordinate, and another pixel with the y coordinate, add the pixel to the array
                if(polarity == 1):
                    array = np.vstack((array, np.array([x_coord, y_coord, 1, 0, 1]))) #Add the pixel to the array
                if(polarity == 0):
                    array = np.vstack((array, np.array([x_coord, y_coord, 0, 1, 1]))) #Add the pixel to the array
        else: #If there is not a pixel with the y coordinate, add the pixel to the array
            if(polarity == 1):
                array = np.vstack((array, np.array([x_coord, y_coord, 1, 0, 1]))) #Add the pixel to the array
            if(polarity == 0):
                array = np.vstack((array, np.array([x_coord, y_coord, 0, 1, 1]))) #Add the pixel to the array
    else: #If the pixel is not in the array, add it to the array
        if(polarity == 1):
            array = np.vstack((array, np.array([x_coord, y_coord, 1, 0, 1]))) #Add the pixel to the array
        if(polarity == 0):
            array = np.vstack((array, np.array([x_coord, y_coord, 0, 1, 1]))) #Add the pixel to the array

    return array

    

def display_pixels(array, matrix, file_path, image_title):
#The function helps to visualize the pixels of an array
#Function to display in a matrix, the pixels of an array
#Both the matrix and the array, are input parameters of the function
#Parameter array : array with the pixels to display
#Parameter matrix : matrix to display the pixels in
#Parameter file_path : location of the data file
#Parameter image_title : title of the image with the pixels in the input array

    for i in range( len( array[:,0] ) ):
        fill_matrix(matrix, array[i, 0], array[i, 1], array[i, 4]) #Put the pixels in the matrix

    max_value_matrix = np.max(array[:, 4])

    #Display the image
    fig1, ax1 = plt.subplots()
   
    pos1 = ax1.imshow(matrix, cmap = 'cividis_r', interpolation = 'none', vmax = max_value_matrix)
    fig1.colorbar(pos1, ax = ax1, shrink = 0.8)
    ax1.set_title(image_title)
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')
    
    #Highlight the pixels in the matrix with an arrow next to the pixel
    annotate_pixels(array, ax1)

    #Display the date, time and file data in the image
    display_extra_info(ax1, file_path)

    #Save the image
    data_name = file_path[55:-4]
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_StarFiltering_' + actual_data_time + '.pdf'
    fig1.savefig(file_image_name, bbox_inches='tight', dpi=600)

    plt.show()

def fill_matrix (matrix, x_coord, y_coord, value):
#Function to put a value in a specific pixel in a matrix
#Parameter matrix : matrix to fill
#Parameter x_coord : coordinate x pixel
#Parameter y_coord : coordinate y pixel

    matrix[y_coord, x_coord] = value

def filter_array(array, value, event_type, condition):
#Function to filter an array by its number of events in a selected column.
#Parameter array: Array to filter
#Parameter value: Value used to filter the events
#Parameter event_type : Choose 1 to filter positive events, 2 to filter negative events, 3 to filter total events. The
#function filter the column number event_type + 1, so if I the idea is to filter the events in the column 6 of the array, the parameter event_type should be 5.
#Parameter condition: Choose 1 to filter greater than (>) value, choose 2 to filter less than (<) value

    if(condition == 1):
        mask = array[:, event_type + 1 ] > value
        return array[mask]
        
    if (condition == 2):
        mask = array[:, event_type + 1 ] < value
        return array[mask]

def zone_histogram(array, bin_width, time_stop, x_coord, y_coord, size_zone, image_title, file_path):
#Function to display a histogram of the pixels of a specific zone of the image and for a specific time duration
#Parameter array : Array with the events 
#Parameter bin_width : Define the bin width for the histogram
#Parameter time_stop : Time limit to display the histogram
#Parameter x_coord : x coordinate of the top leftmost pixel of the desired zone
#Parameter y_coord : y coordinate of the top leftmost pixel of the desired zone
#Parameter size_zone : Size in pixels of one side of the desired zone, e.g. 30x30 zone, size_zone = 30
#Parameter image_title : Beginning of the image title. e.g. if image_title is = 'star', the imag title will be "star Histogram Events"
#Parameter file_path : Variable with the location of the data file
    
    #Filter the events by time
    mask = array[:, -1] <= time_stop
    selectedEvents = array[mask]
    #Filter events by zone of the image
    mask = ( selectedEvents[:, 0] >= x_coord ) & (selectedEvents[:, 0] <= (x_coord + (size_zone - 1)) ) & ( selectedEvents[:,1] >= y_coord ) & (selectedEvents[:,1] <= (y_coord + (size_zone - 1)) )
    selectedEvents = selectedEvents[mask]

    #Variable to know the time of the las event
    timeLastEvent = selectedEvents[-1,-1]
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (timeLastEvent + bin_width), bin_width) # This line works ok for the histogram until the time of last event

    fig, ax = plt.subplots()
    ax.hist(selectedEvents[:,-1], bins=xbins, edgecolor = 'black')

    #plot the xdata locations on the x axis:
    ax.plot(selectedEvents[:,-1], 0*selectedEvents[:,2], 'd', label = 'Data Points')

    plt.title(image_title + " Histogram Events")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Time [us]')
    ax.set_ylabel('Number of events')
    display_extra_info(ax, file_path)
    ax.annotate('Time Bin Width: ' + str(bin_width) + ' us', xy = (0, -41), xycoords = 'axes points', fontsize = 8)
    ax.annotate('Size of the zone [px]: ' + str(size_zone) + 'x' + str(size_zone), xy = (0, -49), xycoords = 'axes points', fontsize = 8)
    ax.annotate('First pixel of the zone: (' + str(x_coord) + ', ' + str(y_coord) + ')', xy = (0, -57), xycoords = 'axes points', fontsize = 8)
    ax.legend()

    data_name = file_path[55:-4]
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_' + image_title + '_ZoneHistogram_' + actual_data_time + '.pdf'
    fig.savefig(file_image_name, bbox_inches='tight', dpi=600)

    plt.show()



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
    display_extra_info(ax)
    ax.annotate('Time Bin Width: ' + str(binWidth) + ' ms', xy = (0, -41), xycoords = 'axes points', fontsize = 8)
    ax.annotate('Size of the zone [px]: ' + str(sizeZone) + 'x' + str(sizeZone), xy = (0, -49), xycoords = 'axes points', fontsize = 8)
    ax.annotate('First pixel of the zone: (' + str(xCoord) + ', ' + str(yCoord) + ')', xy = (0, -57), xycoords = 'axes points', fontsize = 8)
    #plt.show()



def direct_neighbors(array, num_min_events, num_min_neighbors, neighbors, num_column):
#Function to filter pixels by their neighbors pixels with a num_min_events number of events
#If the pixel has num_min_neighbors or plus neighbors, where each neighbor has at least num_min_events events, the pixel will be added to the output array
#Parameter array : The array wih the pixels to filter
#Parameter num_min_events : The minimum number of events in a neighbor pixel to be considered a valid neighbor
#Parameter num_min_neighbors : The minimum number of valid neighbors per pixel to be considered a valid pixel
#Parameter neighbors : The number of neighbors to look for. If neighbors=4, the function will look for neighbors above,
#to the right, below and to the left. If neighbors=8, the function will also look for the neighbors in the diagonals
#Parameter num_column : The number of column in the array to look for the events. e.g. 4=total events, 5=events/time unit

    output_array = np.zeros([1, array.shape[1]], dtype = int) #Output array : x coord, y coord, positive events, negative events, total events, events/time unit
    for i in range( len( array[:,0] ) ) : #Loop through the input array

        num_neighbors = 0 #Variable to check the neighbors number of the pixel

        #Direct neighbors
        #Neighbor above
        pixel = np.where(((array[:, 0]) == (array[i, 0])) & ((array[:, 1]) == ((array[i, 1]) -1 ))) #Find a neighbor pixel in the coordinates above
        if( len(array[pixel]) == 1 ):#There is a neighbor above
            if( array[pixel, num_column] >= num_min_events ):#There is more or equal num_min_events
                num_neighbors += 1 #Neighbor number 1

        #Neighbor right
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == (array[i, 1]))) #Find a neighbor pixel in the coordinates to the right
        if (len(array[pixel]) == 1):  # There is a neighbor to the right
            if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                num_neighbors += 1 # Neighbor number 2

        # Neighbor below
        pixel = np.where( (  (array[:, 0])  ==  (array[i, 0])  ) & (  (array[:, 1])  ==  ((array[i, 1]) + 1)  ) ) #Find a neighbor pixel in the coordinates below
        if (len(array[pixel]) == 1):  # There is a neighbor below
            if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                num_neighbors += 1 # Neighbor number 3

        # Neighbor to the left
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == (array[i, 1])))  #Find a neighbor pixel in the coordinates below
        if (len(array[pixel]) == 1):  # There is a neighbor to the left
            if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                num_neighbors += 1 # Neighbor number 4

        if ( neighbors == 8 ): #Indirect neighbors

            # Neighbor in the upper right side
            pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ( (array[:, 1]) == ((array[i, 1]) - 1)))  # Find a neighbor pixel in the coordinates to the upper right side
            if (len(array[pixel]) == 1):  # There is a neighbor in the upper right side
                if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                    num_neighbors += 1  # Indirect neighbor number 1

            # Neighbor in the lower right side
            pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a neighbor pixel in the coordinates to the lower right side
            if (len(array[pixel]) == 1):  # There is a neighbor to the lower right side
                if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                    num_neighbors += 1  # Indirect neighbor number 2

            # Neighbor in the lower left side
            pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  # Find a neighbor pixel in the coordinates to the lower left side
            if (len(array[pixel]) == 1):  # There is a neighbor to the lower left side
                if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                    num_neighbors += 1  # Indirect neighbor number 3

            # Neighbor in the upper left side
            pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) - 1)))  # Find a neighbor pixel in the coordinates to the upper left side
            if (len(array[pixel]) == 1):  # There is a neighbor to the upper left side
                if (array[pixel, num_column] >= num_min_events):  # There is more or equal num_min_events
                    num_neighbors += 1  # Indirect neighbor number 4

        #Add the pixel if it has the minimum number of neighbors
        if( num_neighbors >= num_min_neighbors ):
            output_array = np.vstack((output_array, array[i]))

    output_array = np.delete(output_array, 0, 0) #Delete the first row because it is 0,0,0,0,0
    return output_array



def is_star(array):
#Function to identify if two or more pixels which are direct neighbors belong to a single star, and select the pixel with most events, as the star pixel
#The function takes every pixel in the input array, looks for all its 8 neighbors and identifies the pixel with the highest number of events as the star pixel
#The function returns an array with all the star pixels
#Parameter array : The array with the pixels to filter

    output_array = np.zeros([1, array.shape[1]], dtype=int)
    
    for i in range(len(array[:, 0])):
        
        star = True #Variable to identify the star pixel. It is assumed that the i pixel has the highest number of events, so that pixel is the star
        
        # Neighbor above
        pixel = np.where(((array[:, 0]) == (array[i, 0])) & ((array[:, 1]) == ((array[i, 1]) - 1)))  #Find a neighbor pixel in the coordinates above
        if (len(array[pixel]) == 1):  # There is a neighbor above
            if (array[pixel, 4] > array[i, 4]):  # The neighbor above has more events
                star = False  # The i pixel is not the star

        # Neighbor right
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == (array[i, 1])))  #Find a neighbor pixel in the coordinates to the right
        if (len(array[pixel]) == 1):  # There is a neighbor to the right
            if (array[pixel, 4] > array[i, 4]):  # The neighbor to the right has more events
                star = False  # The i pixel is not the star

        # Neighbor below
        pixel = np.where(((array[:, 0]) == (array[i, 0])) & ((array[:, 1]) == ((array[i, 1]) + 1)))  #Find a neighbor pixel in the coordinates below
        if (len(array[pixel]) == 1):  # There is a neighbor below
            if (array[pixel, 4] > array[i, 4]):  # The neighbor below has more events
                star = False  # The i pixel is not the star

        # Neighbor to the left
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == (array[i, 1])))  #Find a neighbor pixel in the coordinates to the left
        if (len(array[pixel]) == 1):  # There is a neighbor to the left
            if (array[pixel, 4] > array[i, 4]):  # The neighbor to the left has more events
                star = False  # The i pixel is not the star

        # Neighbor in the upper right side
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == ((array[i, 1]) - 1))) #Find a neighbor pixel in the coordinates to the upper right side
        if (len(array[pixel]) == 1):  # There is a neighbor in the upper right side
            if (array[pixel, 4] > array[i, 4]):  # The neighbor in the upper right side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the lower right side
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) + 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  #Find a neighbor pixel in the coordinates to the lower right side
        if (len(array[pixel]) == 1):  # There is a neighbor in the lower right side
            if (array[pixel, 4] > array[i, 4]):  # The neighbor in the lower right side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the lower left side
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) + 1)))  #Find a neighbor pixel in the coordinates to the lower left side
        if (len(array[pixel]) == 1):  # There is a neighbor in the lower left side
            if (array[pixel, 4] > array[i, 4]):  # The neighbor in the lower left side has more events
                star = False  # The i pixel is not the star

        # Neighbor in the upper left side
        pixel = np.where(((array[:, 0]) == ((array[i, 0]) - 1)) & ((array[:, 1]) == ((array[i, 1]) - 1)))  #Find a neighbor pixel in the coordinates to the upper left side
        if (len(array[pixel]) == 1):  # There is a neighbor in the upper left side
            if (array[pixel, 4] > array[i, 4]):  # The neighbor in the upper left side has more events
                star = False  # The i pixel is not the star

        if (star):
            output_array = np.vstack((output_array, array[i]))  # Add the pixel to the output array

    output_array = np.delete(output_array, 0, 0)  # Delete the first row of the array because it is 0,0,0,0,0
    
    return output_array



def annotate_pixels(array, ax):
#Function to highlight or make more visible the pixels of the input array, in a matrix
#The function draws an arrow next to all the pixels in the input array
#Parameter array : The array with the pixels
#Parameter ax : Variable of the matplotlib figure where the matrix is

    for i in range( len( array[:,0] ) ) :
        ax.annotate(str(i), xy=(array[i,0]+3, array[i,1]), xytext=(array[i,0]+12, array[i,1]), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

##def continuity(array, e):
###Function to analyze along the time an array and determine if here is continuity of events.
###The function look for continuity in the array, that means look for a number of events in the array. The number of events is determined by the parameter e.
###Parameter array : The array to analyze
###Parameter e : The minimum number of events in the array to have continuity
##
##    if( len(array) >= e ):
##        return True
##    return None

##def continuousStar(array, interval, timeStop, level):
###Function to identify a star looking for the continuity of the events in a time interval.
###Parameter array : The array where the possible stars are.
###Parameter interval : The time interval to evaluate the continuity of the events.
###Parameter timeStop : The time limit to look for the continuity of the events.
###Parameter level : The minimum number of evidences to consider a pixel a star.
##
##    windows = np.ceil(timeStop / interval) # Number of time windows to make the filtering
##    outputArray = np.zeros([1, 5], dtype = int) # Output array
##
##    for i in range( len (array[:, 0]) ): # Go through the input array
##        index = np.where( ( events[:, 0] == array[i, 0] ) & ( events[:, 1] == array[i, 1] ) ) # Find pixel events in all the events data
##        onePixelIndexs = np.array(index) # Convert to array the tuple with the indices of the pixel events
##        onePixelEvents = np.take(events, onePixelIndexs, axis=0) # Extract only the events of the pixel from all events data
##        onePixelEvents = np.reshape(onePixelEvents,(onePixelEvents.shape[1],onePixelEvents.shape[2])) # Build the array (nx4) of the pixel events
##        onePixelEvents[: , -1] -= min(onePixelEvents[:,  -1]) # Start the pixel events from t = 0
##
##        proofs = 0
##        for j in range(int(windows)): # Go through the windows or time intervals
##            mask = ( ( (onePixelEvents[:, -1]) >= (j * interval) ) & ( (onePixelEvents[:, -1]) < ( (j + 1) * interval ) ) ) # Select only the events in the time interval
##            pixelsSliced = onePixelEvents[mask] # Array with the events in the time interval
##            # Function to analyze the pixels in the time interval
##            mark = continuity(pixelsSliced, 2)
##            if mark: # If the array in the time interval corresponds to a star
##                proofs = proofs + 1
##        if proofs >= level:
##            outputArray = np.vstack((outputArray, array[i]))
##
##    outputArray = np.delete(outputArray, 0, 0)
##    return outputArray

def calculate_events_per_time(number_total_events, total_time_data, unit_of_time):
#Function to calculate the number of events per unit of time, of one pixel.
#Parameter number_total_events : The total number of events of the pixel.
#Parameter total_time_data : The total time duration of the data.
#Parameter unit_of_time : The time unit reference in microseconds (1000000 for 1 second e.g.) for the events.

    events_per_time = ( unit_of_time * number_total_events ) / ( total_time_data ) #Calculation of the events/time unit
    return events_per_time


def add_events_per_time(array, total_time_data, unit_of_time):
#Function to add the number of events per second (or per miute e.g.), to an array of pixels.
#Parameter array : An array with pixels to add their number of events per amount of time.
#Parameter total_time_data : The total time duration of the data.
#Parameter unit_of_time : The time unit reference in microseconds (1000000 for 1 second e.g.) for the events.

    for i in range( len( array ) ): #Loop through input array
        number_to_add = calculate_events_per_time( array[i, 4], total_time_data, unit_of_time) #Calculate the events/time unit
        array[i, 5] = number_to_add #Add the value of events/time unit to the pixel

    return array

def star_nan(array, matrix):
#Function to fill with NaN terms a star pixel and its 8 neighbors pixels.
#Parameter array : The star to fill with NaN terms, in form of an array with the coordinates in the form = [x,y,...].
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


def neighbor_one(array, matrix):
#Function to fill with 1 terms the 8 neighbor pixels of a star. The star pixel will not be filled with the 1 term.
#Parameter array : The star to fill the neighbors with 1 terms, in form of an array with the coordinates in the form = [x,y,p,t,.].
#Parameter matrix : The matrix to fill with the 1 terms in.

    fill_matrix(matrix, ( array[0] + 1 ), array[1], 1)  #Right neighbor pixel
    fill_matrix(matrix, ( array[0] + 1 ), ( array[1] + 1 ), 1)  #Right-down neighbor pixel
    fill_matrix(matrix, array[0], ( array[1] + 1 ), 1)  #Down neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), ( array[1] + 1 ), 1)  # Left-down neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), array[1], 1)  # Left neighbor pixel
    fill_matrix(matrix, ( array[0] - 1 ), ( array[1] - 1 ), 1)  # Left-up neighbor pixel
    fill_matrix(matrix, array[0], ( array[1] - 1 ), 1)  # Up neighbor pixel
    fill_matrix(matrix, ( array[0] + 1 ), ( array[1] - 1 ), 1)  # Right-up neighbor pixel


def nan_mask(stars_array, input_matrix):
#Function to create a matrix with NaN terms in the star pixels, neighbor star pixels, and a rectangle where the meteor is, using other functions.
#Parameter stars_array : Array with all the stars. Each element in the array should have the form [x,y,.,.,] where x and y are the coordinates of the star.
#Parameter input_matrix : The input matrix where the mask will be.

    for r in range( len( stars_array ) ):
        star_nan( stars_array[r], input_matrix )

    #rectangle_nan(73, 93, 456, 480, input_matrix) # NaN square for the meteor of first file meteor.csv
    rectangle_nan(223, 231, 306, 327, input_matrix) # NaN square for the meteor of second file meteor_003019_long.csv

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


##def noPixelAlone(array):
### Function to identify a pixel alone without neighbors pixels and delete it form an array.
### Parameter array : The array with the pixels to identify the pixels alone of.
##
##    outputArray = np.zeros([1, array.shape[1]], dtype=int)
##    # for i in range( len( array[:, 0] ) ):
##    #     for j in range( 8 ):

##def starCoordinatesList(array):
###Function to make a list with the coordinates of the stars and their 8 direct neighbors, who are in the array input of this function.
###Parameter array : The array with the pixels of the stars.
##
##    outputArray = np.zeros([1,2], dtype=int) #Create the output array.
##
##    for i in range( len(array) ): #Go through the input array.
##        outputArray = np.vstack((outputArray, np.array([array[i,0], array[i,1]]))) #Add the star coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0], array[i, 1] - 1]))) #Add the neighbor above coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1] - 1]))) #Add the upper right side neighbor coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1]]))) #Add the right neighbor coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] + 1, array[i, 1] + 1]))) #Add the lower right side neighbor coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0], array[i, 1] + 1])))  #Add the neighbor below coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1] + 1])))  #Add the lower left side neighbor coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1]])))  # Add the left neighbor coordinate to the output array.
##        outputArray = np.vstack((outputArray, np.array([array[i, 0] - 1, array[i, 1] - 1])))  # Add the upper left side neighbor coordinate to the output array.
##
##    outputArray = np.delete(outputArray, 0, 0) #Delete the first row because it is [0,0].
##    return outputArray

##def meteorCoordinatesList(x1, x2, y1, y2):
###Function to make a list with the coordinates of the rectangle where the meteor's trajectory is.
###Parameter x1 : The x coordinate where the rectangle starts.
###Parameter x2 : The x coordinate where the rectangle ends.
###Parameter y1 : The y coordinate where the rectangle starts.
###Parameter y2 : The y coordinate where the rectangle ends.
##
##    outputArray = np.zeros([1,2], dtype=int) #Create the output array.
##
##    for j in range((y2-y1)+1):
##        for i in range((x2-x1)+1):
##            outputArray = np.vstack((outputArray, np.array([i+x1, j+y1]))) #Add the coordinate to the output array.
##
##    outputArray = np.delete(outputArray, 0, 0)  # Delete the first row because it is [0,0].
##    return outputArray

def histogram_num_events(array, bin_width, file_path):
#Function to display an histogram of the total events of an input array
#This function was made for the analysis of the real sky's pixels
#By default, just the y axis is in log scale. It is necessary to uncomment some lines to have the x axis in log scale too
#The function also saves the image automatically in pdf format
#Parameter array : The array with the pixels to make the histogram of
#Parameter bin_width : The parameter to define the bin width of the histogram in microseconds
#Parameter file_path : Variable with the location of the data file

    #Variable to know the max number of events in the array (events of crazy pixel)
    max_event_array = max(array)
    #Build an array with the bins for the histogram
    xbins = np.arange(0, (max_event_array + bin_width), bin_width) #Bins in linear (normal) scale
    #Build an array with the bins for the histogram in log scale
    bins_log = np.logspace(np.log10(array.min()), np.log10(array.max()), 40) #Modify the number 70 to change the scale
    
    #Create the figure
    fig, ax = plt.subplots()
    ax.hist(array[:], bins = xbins, edgecolor = 'orange') #x axis in log scale -> bins = bins_log
    plt.yscale('log') #y axis in log scale
    #plt.xscale('log') #Uncomment to have x axis in log scale
    #plot the xdata locations on the x axis:
    ax.plot(array[:], 0*array[:], 'd')

    plt.title("Histogram of Total Events : Real Sky")
    plt.grid(visible = True, color = 'r')
    ax.set_xlabel('Events')
    ax.set_ylabel('Number of pixels')
    display_extra_info(ax, file_path)
    ax.annotate('Events Bin Width: ' + str(bin_width) + ' events', xy = (0, -46), xycoords = 'axes points', fontsize = 8)

    #To save the image
    data_name = file_path[55:-4]
    actual_data_time = str(datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
    file_image_name = data_name + '_HistoRealSky_' + actual_data_time + '.pdf'
    fig.savefig(file_image_name,bbox_inches='tight', dpi=600)
    
    plt.show()
