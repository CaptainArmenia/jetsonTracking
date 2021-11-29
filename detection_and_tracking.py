#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys

from sort.sort import *
import cv2

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="",
                    nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf",
                    help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video output object
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# create tracker
mot_tracker = Sort(max_age=30, 
                       min_hits=20,
                       iou_threshold=0.2)

frame_count = 0
#cap = cv2.VideoCapture(opt.input_URI)
#ret, frame = cap.read()
#input_shape = (640, 360)

# font
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 0.5
tracker_color = (0, 0, 255)
detection_color = (0, 255, 0)
# Line thickness of 2 px
thickness = 2

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()
    #frame = cv2.resize(frame, input_shape)
    #img = jetson.utils.cudaFromNumpy(frame)

    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay="none")

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        print(detection)

    if len(detections) == 0:
        detections = np.empty((0, 5))
    else:
        detections = np.array([[x.Left, x.Top, x.Right, x.Bottom, 1] for x in detections])

    track_bbs_ids = mot_tracker.update(detections)
    
    cpu_img = jetson.utils.cudaToNumpy(img)
    
    for detection in detections:
        rect_start = (int(detection[0]), int(detection[1]))
        rect_end = (int(detection[2]), int(detection[3]))
        cpu_img = cv2.rectangle(cpu_img, rect_start, rect_end, detection_color, thickness)

    for tracker in mot_tracker.trackers:
        bbox = tracker.get_state()
        rect_start = (int(bbox[0][0]), int(bbox[0][1]))
        rect_end = (int(bbox[0][2]), int(bbox[0][3]))
        center = (int((bbox[0][2] + bbox[0][0]) / 2), int((bbox[0][3] + bbox[0][1]) / 2))
        text_coords = (int((bbox[0][2] + bbox[0][0]) / 2) + 10, int((bbox[0][3] + bbox[0][1]) / 2))
        #cpu_img = cv2.rectangle(cpu_img, rect_start, rect_end, tracker_color, thickness)
        cpu_img = cv2.circle(cpu_img, center, 5, tracker_color, thickness)
        cpu_img = cv2.putText(cpu_img, str(tracker.id), text_coords, font, fontScale, tracker_color, thickness, cv2.LINE_AA)

    #if frame_count == 36:
    #    print("stop")
    #if len(track_bbs_ids) == 0 and frame_count > 150:
    #    print("no trackers")

    frame_count += 1

    cv2.imshow('Frame',cv2.cvtColor(cpu_img, cv2.COLOR_RGB2BGR))
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("stopping")
        break


    # render the image
    #output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(
        opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()
    
    # exit on input/output EOS
    if not input.IsStreaming():
        break
    #ret, frame = cap.read()

#cap.release()

# Closes all the frames
cv2.destroyAllWindows()

