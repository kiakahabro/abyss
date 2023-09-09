#! /usr/bin/env python
import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import message_filters
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()


def rot_x(theta: float) -> np.ndarray:
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def rot_y(theta: float) -> np.ndarray:  
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])


def rot_z(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def euler_to_rot(euler_angles: np.ndarray) -> np.ndarray:
    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]
    Rnc = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return Rnc


def blend_warped_images(warped_images):
    # Determine the dimensions of the panorama canvas
    panorama_width = warped_images[0].shape[1]
    panorama_height = warped_images[0].shape[0]


    # Create a blank panorama canvas
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)


    # Blend each warped image onto the panorama canvas
    for warped_img in warped_images:
        assert panorama_width == warped_img.shape[1], "All warped images must have the same width"
        assert panorama_height == warped_img.shape[0], "All warped images must have the same height"

        # img2gray = cv2.cvtColor(warped_img,cv2.COLOR_BGR2GRAY)
        # # Find the non-zero pixels in the warped image
        # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

        # mask_inv = cv2.bitwise_not(mask)
        # # Blend the warped image onto the panorama canvas using the binary mask
        # panorama_keep   = cv2.bitwise_and(panorama, panorama, mask=mask_inv)
        # panorama_update = cv2.bitwise_and(warped_img, warped_img, mask=mask)
        # panorama = cv2.add(panorama_keep, panorama_update)
        mask = cv2.bitwise_or(warped_img[:, :, 0], warped_img[:, :, 1])
        mask = cv2.bitwise_or(mask, warped_img[:, :, 2])

        # Blend the warped image onto the panorama canvas using the binary mask
        panorama= cv2.bitwise_and(panorama, panorama, mask=cv2.bitwise_not(mask)) + warped_img

    return panorama


class Camera:
    def __init__(self, image_topic, camera_name, cal_filepath: Path) -> None:
        self.image_topic = image_topic
        self.camera_name = camera_name
        cal_filepath = Path(cal_filepath)
        if not cal_filepath.exists():
            raise FileNotFoundError(f"Could not find calibration file for {camera_name} at {cal_filepath}")
        self.cal_filepath = cal_filepath

        with open(cal_filepath) as f:
            cal_data = json.load(f)
        assert isinstance(camera_name, str), "Expected camera_name to be a str object"
        
        data = cal_data[camera_name]

        # Get intrinsics
        intrinsics = data["intrinsics"]
        image_size = intrinsics["image_size"]
        distortion = intrinsics["distortion"]
        principle_point = intrinsics["principal_point"]
        focal_length = intrinsics["focal_length"]

        # Get image dimensions
        self.width = image_size["x"]
        self.height = image_size["y"]
        self.focal_length = focal_length

        self.K = np.array([[focal_length, 0, principle_point["x"]],
                           [0, focal_length, principle_point["y"]],
                           [0, 0, 1]])
        
        radial = distortion["radial"]
        tangential = distortion["tangential"]
        self.distortion_coeffs = np.array([radial["k1"], radial["k2"], tangential["p1"], tangential["p2"], radial["k3"]])

        # Get extrinsics
        extrinsics = data["extrinsics"]
        pose = extrinsics["pose"]
        euler_angles = np.array((pose["rotation"]["roll"], pose["rotation"]["pitch"], pose["rotation"]["yaw"]))

        # Convert euler angles to rotation matrix
        self.Rnc = euler_to_rot(euler_angles)

        # Get translation vector
        self.rCNn = np.array([pose["translation"]["x"], pose["translation"]["y"], pose["translation"]["z"]])
        self.rCNn = self.rCNn.reshape((3, 1))


    def getImageTopic(self) -> str:
        return self.image_topic
    

    def getPosition(self) -> np.ndarray:
        return self.rCNn
    

    def getRotation(self) -> np.ndarray:
        return self.Rnc
    

    def getCameraMatrix(self) -> np.ndarray:
        return self.K
    

    def getDistortionCoeffs(self) -> np.ndarray:
        return self.distortion_coeffs
    

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.K, self.distortion_coeffs)
    

    def getProjectionMatrix(self) -> np.ndarray:
        PI = np.hstack((self.Rnc, self.rCNn))
        return self.K @ PI
    

    def __repr__(self) -> str:
        out = f"Camera({self.camera_name}) [{self.width} x {self.height}]\n"
        out += f"Image topic: {self.image_topic}\n"
        out += f"Position: {self.rCNn}\n"
        out += f"Rotation: \n{self.Rnc}\n"
        return out


class ImageFusion:
    def __init__(self, cameras: List[Camera], pub_topic: str) -> None:
        self.cameras = cameras
        self.pub_topic = pub_topic
        self.bridge = CvBridge()

        # Publisher
        self.pub = rospy.Publisher(self.pub_topic, Image, queue_size=10)

        # Subscribes to all topics and synchronizes them
        self.subs = [message_filters.Subscriber(camera.getImageTopic(), Image) for camera in self.cameras]
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        rCNn_all = np.asarray([camera.getPosition() for camera in self.cameras])
        rCNn_mean= rCNn_all.mean(axis=1)
        err = rCNn_mean - rCNn_all.squeeze()
        nerr = np.linalg.norm(err, axis=0)
        central_cam_idx = np.argmin(nerr)

        self.rPNn = cameras[central_cam_idx].getPosition()
        self.Rnp = cameras[central_cam_idx].getRotation()

        
        
    def calculateHomography(self, rCNn: np.ndarray, Rnc: np.ndarray) -> np.ndarray:
        # Calculate the homography matrix

        rCPp = self.Rnp.T @ (rCNn - self.rPNn)
        Rpc = self.Rnp.T @ Rnc
        npn  = -self.Rnp[:, [2]]
        d = 1
        denom = (d + npn.T @ rCNn)
        
        Hpc = Rpc - rCPp @  ((npn.T @ Rnc) / denom)
        return Hpc


    def callback(self, *image_msgs: Tuple[Image]):
        images = []
        for img_msg in image_msgs:
            if img_msg.encoding != "bgr8":
                raise ValueError(f"Expected image encoding to be bgr8, but got {img_msg.encoding}")
            images.append(self.bridge.imgmsg_to_cv2(img_msg, "bgr8"))
            
        fused = self.fuseByStacking(images, [2, 1, 0])
        fused_msg = self.bridge.cv2_to_imgmsg(fused, "bgr8")
        self.pub.publish(fused_msg)


    def fuseByStacking(self, images: Any, stack_order: List[int] = None) -> Any:
        if stack_order is None:
            return cv2.hconcat(images)
        else:
            assert len(images) == len(stack_order), "Expected number of images to be equal to number of cameras"
            image_sort = [images[i] for i in stack_order]
            return cv2.hconcat(image_sort)


    def fuseFromExtrinsics(self, images: Any) -> Any:
        assert len(images) == len(self.cameras), "Expected number of images to be equal to number of cameras"
        
        undistorted_images = [self.cameras[i].undistort(images[i]) for i in range(len(images))]
        panorama_width  = sum([ud_img.shape[1] for ud_img in undistorted_images]) 
        panorama_height = max([ud_img.shape[0] for ud_img in undistorted_images]) 
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

        warped_imgs = []
        for i, img in enumerate(undistorted_images):
            H = self.calculateHomography(self.cameras[i].getPosition(), self.cameras[i].getRotation())
            warped_img = cv2.warpPerspective(img, H, (panorama_width, panorama_height))
            warped_imgs.append(warped_img)
        
        panorama = blend_warped_images(warped_imgs)
            
        return panorama


if __name__ == '__main__':
    rospy.init_node("image_fusion_node")

    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        
        # Get calibration data
        calibration_file = Path(os.path.join(rospack.get_path('abyss')), "data", "intrinsics_extrinsics.json")
        if not calibration_file.is_file():
            raise FileNotFoundError(f"Could not find calibration file at {calibration_file}")
        
        # Build Cameras object
        camera_names = ["camera_1", "camera_2", "camera_3"]
        cameras = [Camera(f"/platypus/{camera}/dec/manual_white_balance", camera, calibration_file) for camera in camera_names]
        pub_topic = "/fused_image"
        
        img_fusion = ImageFusion(cameras, pub_topic)
        print("Image fusion node started")
        rospy.spin()
    except rospy.ROSInterruptException: 
        pass