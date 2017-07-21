import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import time
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split



# Convert .png(float32, bin range 0~1) to jpg(uint8, bin range 0~256)
# use this function if needed
def convert_png_2_jpg(img):    
    imag = img * 255
    image = imag.astype(np.uint8)
    return image


	
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)




def pickle_image_folder(folder,pickle_name,cars):
    for n in range(len(cars_files)):
        pngs =  glob.glob((folder + '/**/*.png'), recursive=True)
        jpgs =  glob.glob((folder + '/**/*.jpg'), recursive=True)
        
    imgs = []
    for file in pngs:
        # Read in each one by one
        image = mpimg.imread(file)
        image = image * 255
        image = image.astype(np.uint8)
        imgs.append(image)
        
    for file in jpgs:
        image = mpimg.imread(file)
        imgs.append(image)
        
    if cars == 'yes':
        # Define the labels vector
        y = np.hstack(np.ones(len(imgs)))
    elif cars == 'no':
        y = np.hstack(np.zeros(len(imgs)))
    else:
        print('Declare cars or not!')
        
    # save scaler as pickled file
    pickle.dump({'X':imgs,'y':y},open(pickle_name,"wb"))
        

		
		


    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def add_heat(heatmap,win_list):
    # Iterate through list of windows
    for win in win_list:
        # Add +=1 for all pixels inside each win
        # assuming each win takes the form ((x1,y1),(x2,y2))
        heatmap[win[0][1]:win[1][1],win[0][0]:win[1][0]] += 1
        
    return heatmap




# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    else:
        return img
    



def draw_labeled_windows(img, labels):
    # Iterate through all detected cars
    for car_num in range(1, labels[1]+1):
        # Find pixels with each car_num label value
        nonzero = (labels[0] == car_num).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        win = ((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, win[0], win[1], (255,0,0), 6)

    # Return image
    return img


def add_heat(heatmap,win_list):
    # Iterate through list of windows
    for win in win_list:
        # Add +=1 for all pixels inside each win
        # assuming each win takes the form ((x1,y1),(x2,y2))
        heatmap[win[0][1]:win[1][1],win[0][0]:win[1][0]] += 1
        
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # return thresholded map
    return heatmap


def draw_labeled_windows(img, labels):
    # Iterate through all detected cars
    for car_num in range(1, labels[1]+1):
        # Find pixels with each car_num label value
        nonzero = (labels[0] == car_num).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        win = ((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, win[0], win[1], (255,0,0), 6)

    # Return image
    return img


def test_image(image,colorspace,orient,pix_per_cell,cell_per_block,point_scale_data):
    img = np.copy(image)
    #colorspace = "YCrCb"
    windows = search_with_multiscale_windows(image, colorspace, orient, pix_per_cell, cell_per_block,
                                             point_scale_data)
    
    if len(windows) > 0:
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, windows)
        # Lower value increases likelyhood of drawing box
        heat = apply_threshold(heat,10)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        draw_img = draw_labeled_windows(img, labels)
    else:
        print("No windows found")
        img = image
    
    return draw_img,heatmap




'''
Below are moved functions from main
'''

def bin_spatial(img, size=(32, 32)):
    #print('shape of image is: ' + str(np.shape(img)))
    #print('spatial size is: ' + str(spatial_size))
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, y_start, y_stop, scale, colorspace, hog_channel, svc, X_scaler, orient, 
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    
    # array of rectangles where cars were detected
    rectangles = []
    #if img.dtype == 'float32':
    #    img = convert_png_2_jpg(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[y_start:y_stop,:,:]

    # apply color conversion if other than 'RGB'
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
            #print('Changed to YCrBc')
    else: ctrans_tosearch = np.copy(image)   
    
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    global spatial_features,hog_features,hist_features
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            ################ ONLY FOR BIN_SPATIAL AND COLOR_HIST ################

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

            # Combine the various features to one array
            stacked_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            #print('mean test_features before is ' + str(np.mean(stacked_features)))


            

            # Scale features and make a prediction
            test_features = X_scaler.transform(stacked_features)    
            #print('mean test_features scaled is ' + str(np.mean(test_features)))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            ######################################################################
            
            #test_prediction = svc.predict(hog_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+y_start),(xbox_left+win_draw,ytop_draw+win_draw+y_start)))
                
    return rectangles
	
	
# Define a function to extract features from a list of image locations
# This function could also be used to call bin_spatial() and color_hist() (as in the lessons) to extract
# flattened spatial color features and color histogram features and combine them all (making use of StandardScaler)
# to be used together for classification
def extract_features(imgs, colorspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32,32), hist_bins = 16):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if colorspace != 'RGB':
            if colorspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif colorspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif colorspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif colorspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif colorspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            elif colorspace == 'Lab':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        else: feature_image = np.copy(image)      

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
        
        # Apply color_hist()  
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)    
            
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            
        # Add HOG third    
        file_features.append(hog_features)
        # Add all three features to list of all images
        features.append(np.concatenate(file_features))
        
    # Return list of feature vectors
    return features



def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features



# Function courtesy of TMresolution0115
def pickle_extracted_features(pickle_file, cars, notcars, colorspace, orient, 
                              pix_per_cell,cell_per_block, hog_channel,
                              spatial_size=(32,32), hist_bins=32):
    print('pickle ext colorspace is ',colorspace)
    # Time Start
    time1 = time.time()
    # extract each features
    car_features = extract_features(cars, colorspace, orient, 
                        pix_per_cell, cell_per_block, 
                        hog_channel, spatial_size, hist_bins)
    notcar_features = extract_features(notcars, colorspace, orient, 
                        pix_per_cell, cell_per_block, 
                        hog_channel, spatial_size, hist_bins)
    # Time End
    time2 = time.time()
    print(' Seconds to extract features: ',round(time2-time1, 2))
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    global X_scaler
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    print('before scale cars ' + str(np.mean(car_features)) + 'and length ' + str(np.shape(car_features)))
    print('before scale' + str(np.mean(X)))
    print('after  ' + str(np.mean(scaled_X)))
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))
    
    # save scaler as pickled file
    pickle.dump({'X_scaler':X_scaler,'scaled_X':scaled_X,'y':y},open(pickle_file,"wb"))
    
    print("The input was: colorspace = {}, orient = {},\
    pix_per_cell = {}, cell_per_block = {}".format(colorspace,orient,pix_per_cell,cell_per_block) )
    print("Extracted features are stored in pickled file:{}".format(pickle_file))


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def bin_spatial(img, size=(32,32)):
    #print('shape of image is: ' + str(np.shape(img)))
    #print('spatial size is: ' + str(size))
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def search_with_multiscale_windows(img, colorspace, orient, pix_per_cell, cell_per_block, 
                                   point_scale_data, hog_channel, spatial_size=(32,32), hist_bins=32):
    #print('search_with_multiscale colorspace is ',colorspace)
    # Define array to store recent windows
    windows = []
    origin = np.copy(img)
    svc = pickle.load(open('svc.pickle','rb'))
    
    scal_data = pickle.load(open('X_scaler_scaled_X_y.p','rb'))
    X_scaler = scal_data["X_scaler"]
    
    y_start=point_scale_data[0]
    y_stop=point_scale_data[1]
    scale=point_scale_data[2]
    #print(y_start,y_stop,scale)
    
    for y_start,y_end, scales in zip(y_start,y_stop,scale):
        #print(y_start)
        rectangles = find_cars(img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins)
        
        #windows = windows + win_list
    
    return rectangles


def process_frame(img, colorspace, y_start, y_stop, scale, orient, pix_per_cell, 
                  cell_per_block, point_scale_data, hog_channel, spatial_size, hist_bins,
                  X_scaler, svc):

    rects = []

    test_img = img
    #colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #orient = 11
    #pix_per_cell = 16
    #cell_per_block = 2
    #hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    y_start = 400
    y_stop = 464
    scale = 1.0
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 416
    y_stop = 480
    scale = 1.0
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 400
    y_stop = 496
    scale = 1.5
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 432
    y_stop = 528
    scale = 1.5
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 400
    y_stop = 528
    scale = 2.0
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 400
    y_stop = 596
    scale = 3.5
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))
    y_start = 464
    y_stop = 660
    scale = 3.5
    rects.append(find_cars(test_img, int(y_start), int(y_stop), scale, colorspace, 
                       hog_channel, svc, X_scaler, orient, pix_per_cell, 
                       cell_per_block, spatial_size, hist_bins))

    rectangles = [item for sublist in rects for item in sublist] 
    
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    draw_img, rects = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image and final rectangles
    return img, rects


def process_video(img, colorspace, y_start, y_stop, scale, orient, pix_per_cell, 
                  cell_per_block, point_scale_data, hog_channel, spatial_size, hist_bins,
                  X_scaler, svc):
    cimg = np.copy(image)
    windows = search_with_multiscale_windows(image, colorspace, orient, pix_per_cell, cell_per_block,
                                             point_scale_data, hog_channel, spatial_size, hist_bins)
    

    iteration = 1
    #print("iterations: ",iteration)
    #global iteration, heatmaps
    if len(windows) > 0:
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, windows)
        heatmaps.append(heat)
        #print(len(heatmaps))
        # take recent 10 heatmaps and average them
        if len(heatmaps) == 3:
            avg_heat = sum(heatmaps)/len(heatmaps)
            heat = avg_heat
        heat = apply_threshold(heat,1)
        heatmap = np.clip(heat, 0, 255)
        #heatmaps.append(heatmap)
        #if iteration % 10 == 0:
        #    heatmap = avg_heatmaps
        labels = label(heatmap)
        draw_img, rect = draw_labeled_bboxes(cimg, labels)
    else:
        # pass the image itself if nothing was detected
        draw_img = cimg
    iteration += 1
    return draw_img

	
# Define a class to store data from video
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # throw out oldest rectangle set(s)
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]




def process_video_2(img):

    rects = []
    
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    cspace = colorspace # sometimes functions uses this abreviation
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size=(32,32)
    hist_bins = 16 # Errors out when trying to modify
    

  
    # Load the data and trained model weights
    fname = 'X_scaler_scaled_X_y.p'
    try:
        scaled_data
    except NameError:
        scaled_data = pickle.load(open(fname,'rb'))
    else:
        pass
    try:
        svc
    except NameError:
        svc = pickle.load(open('svc.pickle','rb'))
    else:
        pass
    
    X_scaler = scaled_data["X_scaler"]
    
    
    test_img = img

    y_start = (400,416,400,432,400,432,400,464)
    y_stop = (464,480,496,528,528,560,596,660)
    scale = (1.0,1.0,1.5,1.5,2.0,2.0,3.5,3.5)

    for y_start,y_end, scales in zip(y_start,y_stop,scale):
        rects.append(find_cars(img, int(ystart), int(ystop), scale, colorspace, 
                     hog_channel, svc, X_scaler, orient, pix_per_cell, 
                     cell_per_block, spatial_size, hist_bins))

    rectangles = [item for sublist in rects for item in sublist] 

    # add detections to the history
    det = Vehicle_Detect()

    if len(rectangles) > 2:
        det.add_rects(rectangles)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in det.prev_rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)
     
    labels = label(heatmap_img)
    draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

