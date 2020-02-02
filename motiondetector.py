# Python program to implement
# WebCam Motion Detector
from imutils.video import VideoStream
from imutils.video import FPS
import pandas
import numpy as np
import argparse
import imutils
import time
import cv2
import ffmpeg

def FaceBounder():
    # Assigning our static_back to None (background)
    static_back = None
    # List when any moving object appear
    motion_list = [None, None]
    # Time of movement
    time = []

    # Initializing DataFrame, one column is start
    # time and other column is end time
    df = pandas.DataFrame(columns=["Start", "End"])

    video = cv2.VideoCapture('output.avi')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))


    # Infinite while loop to treat stack of image as video
    while (True):
        # Reading frame(image) from video
        ret, img = video.read()
        print(img)
        if ret is True:
            cv2.imshow('test', img)
        else:
            print("I am type None")
        # Initializing motion = 0(no motion)
        motion = 0

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur
        # so that change can be find easily

        gray = cv2.GaussianBlur(gray, (21, 21), 0)


        # In first iteration we assign the value
        # of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Difference between static background
        # and current frame(which is GaussianBlur)
        diff_frame = cv2.absdiff(static_back, gray)

        # If change in between static background and
        # current frame is greater than 30 it will show white color(255)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        (cnts, hierarchy) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for contour in cnts:
            #x is: img
            #y: is color -> gray,
            #w is: Parameter specifying how much the image size is reduced at each image scale.
            #h is: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            for (x,y,h,w) in faces:

                if cv2.contourArea((x,y,h,w)) < 10000:
                    continue
                motion = 1
                #(x,y,w,h) = cv2.boundingRect(x)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                roi = img[y:y + h, x:x + w]

        out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

        roi = cv2.resize(roi, (400, 400), interpolation = cv2.INTER_LINEAR)
        out.write(roi)
        cv2.imshow('img', roi)

        motion_list.append(motion)

        motion_list = motion_list[-2:]

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now())

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now())

        key = cv2.waitKey(1)
        # if q entered whole process will stop
        if key == ord('q'):
            # if something is moving then it append the end time of movement
            if motion == 1:
                time.append(datetime.now())
            break

    video.release()
    out.release()

    cv2.destroyAllWindows()

def CreateVidFromStream():
    capture_duaration = 5
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    start = time.time()
    while(int(time.time() - start)<capture_duaration):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,1)

            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return out

def main():
    vid = CreateVidFromStream()
    FaceBounder()

if __name__ == '__main__':
    main()
