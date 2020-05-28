# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:08:38 2020

@author: Nishidh Shekhawat

Multi-Person Pose Estimation model

2 Pre trained models by open pose : https://github.com/CMU-Perceptual-Computing-Lab/openpose
COCO Dataset : http://cocodataset.org/#download
MPII Human Pose Dataset : http://human-pose.mpi-inf.mpg.de/#download
"""
#%%
import cv2
from tqdm import tqdm 
import download_file as df

#%% For MPII
url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel'
filename ='models\pose_iter_160000.caffemodel'
print("Downloading Pretrained MPII model")
df.downloadfile(url,filename)

protoFile = "models\pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "models\pose_iter_160000.caffemodel"

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
             ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
             ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
             ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

#%% For COCO
url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel'
filename ='models\pose_iter_440000.caffemodel'
print("Downloading Pretrained COCO model")
df.downloadfile(url,filename)

protoFile = "models\pose_deploy_linevec.prototxt"
weightsFile = "models\pose_iter_440000.caffemodel"


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }


POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
                
#%%

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
#print(net.getLayerNames())

# Read image
frame = cv2.imread("images\sample.jpg")

# Specify the input image dimensions
inWidth = 368
inHeight = 368

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), 
                                swapRB=False, crop=False)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output= net.forward()

frameHeight = frame.shape[0]
frameWidth = frame.shape[1]
H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected keypoints
points = []
for i in tqdm(range(len(BODY_PARTS))):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > 0.2 :
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)

for pair in tqdm(POSE_PAIRS):
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]

    if points[idFrom] and points[idTo]:
        cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

t, _ = net.getPerfProfile()
freq = cv2.getTickFrequency() / 1000
cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

cv2.imshow('Output', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()