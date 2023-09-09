#! /usr/bin/env python
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

if __name__ == '__main__':
    print("Image fusion node started")