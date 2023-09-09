#! /usr/bin/env python
from typing import Any, List

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageFusion:
    def __init__(self, sub_topics: List[str], pub_topic: str) -> None:
        self.sub_topics = sub_topics
        self.pub_topic = pub_topic
        self.bridge = CvBridge()

        # Publisher
        self.pub = rospy.Publisher(self.pub_topic, Image, queue_size=10)

        # Subscribes to all topics and synchronizes them
        self.subs = [message_filters.Subscriber(topic, Image) for topic in self.sub_topics]
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback(self, *args):
        images = [self.bridge.imgmsg_to_cv2(img, "bgr8") for img in args]
        fused = self.fuse(images)
        fused_msg = self.bridge.cv2_to_imgmsg(fused, "bgr8")
        self.pub.publish(fused_msg)
    
    def fuse(self, images: Any) -> Any:
        return cv2.hconcat(images)


if __name__ == '__main__':
    rospy.init_node("image_fusion_node")

    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        # Topics to fuse
        sub_topics = ["/platypus/camera_1/dec/manual_white_balance", 
                    "/platypus/camera_2/dec/manual_white_balance",
                    "/platypus/camera_3/dec/manual_white_balance"]
        
        pub_topic = "/fused_image"
        
        img_fusion = ImageFusion(sub_topics, pub_topic)
        print("Image fusion node started")
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass