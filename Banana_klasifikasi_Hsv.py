import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
from skimage import io
training=[]
testing=[]
testing_klasifikasi=[]
training_klasifikasi=[]

for filename in sorted(glob('/home/reizha/Downloads/banana/test/test_matang/*.jpg'), key=lambda name: int(name[-5:-4])):
    x=0     
    if x<40:
        img = cv.imread(filename)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        data_dominant=np.transpose(dominant[0:3,np.newaxis])
        testing.extend(data_dominant)
        testing_klasifikasi.append(1.0)
        data_dominant_testing=pd.DataFrame(testing).astype(np.float32)

for filename in sorted(glob('/home/reizha/Downloads/banana/test/test_mentah/*.jpg'), key=lambda name: int(name[-5:-4])):
    x=0     
    if x<40:
        img = cv.imread(filename)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        data_dominant=np.transpose(dominant[0:3,np.newaxis])
        testing.extend(data_dominant)
        testing_klasifikasi.append(0.0)
        data_dominant_testing=pd.DataFrame(testing).astype(np.float32)
        
for filename in sorted(glob('/home/reizha/Downloads/banana/matang/*.jpg'), key=lambda name: int(name[-5:-4])):
    x=0     
    if x<40:
        img = cv.imread(filename)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        data_dominant=np.transpose(dominant[0:3,np.newaxis])
        training.extend(data_dominant)
        training_klasifikasi.append(1.0)
        data_dominant_training=pd.DataFrame(training).astype(np.float32)
        
for filename in sorted(glob('/home/reizha/Downloads/banana/mentah/*.jpg'), key=lambda name: int(name[-5:-4])):
    x=0     
    if x<40:
        img = cv.imread(filename)
        pixels = np.float32(img.reshape(-1, 3))
        n_colors = 5
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        data_dominant=np.transpose(dominant[0:3,np.newaxis])
        training.extend(data_dominant)
        training_klasifikasi.append(0.0)
        data_dominant_training=pd.DataFrame(training).astype(np.float32)
        
        
from sklearn.naive_bayes import MultinomialNB
model_mulNB = MultinomialNB()
model_mulNB.fit(data_dominant_training, training_klasifikasi)

predicted_mulNB = model_mulNB.predict(data_dominant_training)
print("Predicted Value: ", predicted_mulNB)
print()
response = np.concatenate([training_klasifikasi, predicted_mulNB]).astype(np.float32)
Dataset = np.concatenate([data_dominant_training, data_dominant_training]).astype(np.float32)
from sklearn.model_selection import cross_val_score
score= cross_val_score(model_mulNB, Dataset, response, cv=2, scoring='accuracy')
print("Cross Validation : ", score.mean())




from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1) #define K=3
knn.fit(data_dominant_training,training_klasifikasi)
res1 = knn.predict(data_dominant_testing)
response1 = np.concatenate([training_klasifikasi, res1]).astype(np.float32)
Dataset1 = np.concatenate([data_dominant_training, data_dominant_testing]).astype(np.float32)
angka= cross_val_score(knn, Dataset, response, cv=2, scoring='accuracy')
print(angka)
        

        
        