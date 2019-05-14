#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:14:20 2019

@author: akshit
"""

import numpy as np
import cv2

cam = cv2.VideoCapture(0)

face_cas = cv2.CascadeClassifier('/home/akshit/haarcascade_frontalface_default.xml')

data = []
ix = 0	

while True:
	
	ret, frame = cam.read()

	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		
		for (x, y, w, h) in faces:

			face_component = frame[y:y+h, x:x+w, :]

			fc = cv2.resize(face_component, (50, 50))

			if ix%10 == 0 and len(data) < 20:
				data.append(fc)

			
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		ix += 1
		cv2.imshow('frame', frame)	
		
		key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

cv2.destroyAllWindows()

data = np.asarray(data)

print data.shape

np.save('akshit', data)
