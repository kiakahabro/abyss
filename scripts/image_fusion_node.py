#! /usr/bin/env python
from typing import Any, List

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