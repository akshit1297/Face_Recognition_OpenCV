#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:37:11 2019

@author: akshit
"""

import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/home/akshit/haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

f_01 = np.load('/home/akshit/akshit.npy').reshape((20, 50*50*3))	
f_02 = np.load('/home/akshit/preeti.npy').reshape((20, 50*50*3))

names = {
	0: 'Akshit',
	1: 'Preeti', 
}	

labels = np.zeros((40, 1))
labels[:20, :] = 0.0	
labels[20:, :] = 1.0

data = np.concatenate([f_01, f_02])

skip = 0 
dataset_path = "/home/akshit/"
face_data = []

#KNN

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

#KNN end

while True:
	ret, frame = cap.read()

	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		for (x, y, w, h) in faces:
			face_component = frame[y:y+h, x:x+w, :]
			fc = cv2.resize(face_component, (50, 50))

			lab = knn(fc.flatten(), data, labels)

			text = names[int(lab)]

			cv2.putText(frame, text, (x, y), font, 1, (255, 255, 0), 2)

			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.imshow('face recognition', frame)
        
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

cv2.destroyAllWindows()
