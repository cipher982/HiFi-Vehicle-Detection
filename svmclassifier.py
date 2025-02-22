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

def training_with_SVM(pickleFile):
    # read features from pickled data
    pickled_data = pickle.load(open(pickleFile,"rb"))
    X_scaler = pickled_data["X_scaler"]
    scaled_X = pickled_data["scaled_X"]
    y = pickled_data["y"]
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.3,
                                                        random_state=rand_state)
    
    # Use a linearSVC
    svc = LinearSVC()
    
    # Check the training time for SVC
    # set timer
    time1 = time.time()
    svc.fit(X_train, y_train)
    time2 = time.time()
    print(round(time2 - time1, 2), 'Seconds to train SVC')
    # Check the score of the SVC
    print("Test Accuracy of SVC = ", round(svc.score(X_test, y_test), 4))
    
    with open('svc.pickle','wb') as f:
        pickle.dump(svc,f)
    print('Model saved as [ {} ] file'.format(f.name))
    
    n_predict = 10
    print('My SVC predictions: \n', svc.predict(X_test[0:n_predict]))
    print('For ', n_predict, 'labels: \n', y_test[0:n_predict])