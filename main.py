import cv2
from cv2 import dnn_superres

sr = dnn_superres.DnnSuperResImpl_create()

path = 'EDSR_x4.pb'
sr.readModel(path)
sr.setModel('edsr', 4)

# CUDA
sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

image = cv2.imread('test.png')
upscaled = sr.upsample(image)
cv2.imwrite('upscaled_test.png', upscaled)
