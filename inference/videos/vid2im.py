import cv2
import os
  
# Read the video from specified path
curdir = "/data/Joshua/ASTERS/ASTERS/Computer Vision/inference/videos/"

# cap = cv2.VideoCapture(curdir+"Nested Sequence 05.mp4")
cap = cv2.VideoCapture(curdir+"mall_02.mp4")
# cap = cv2.VideoCapture("output.mp4")

print(cap)
  
try:
      
    # creating a folder named data
    if not os.path.exists(curdir+'data'):
        os.makedirs(curdir+'data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
currentframe = 1
# print('here')  
while(True):
    # print('here')  
    # reading from frame
    ret,frame = cap.read()
    # print(ret, frame)
    if ret:
        # if video is still left continue creating images
        frame_name = f'{currentframe:06d}.jpg'

        name = curdir+'/data/' + frame_name
        print ('Creating...' + name)
  
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break
  
# Release all space and windows once done
cap.release()
cv2.destroyAllWindows()