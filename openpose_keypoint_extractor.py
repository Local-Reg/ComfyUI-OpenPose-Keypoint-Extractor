import json
from nodes import MAX_RESOLUTION

import torch
import numpy as np
import cv2

class OpenPoseKeyPointBase:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "image_width": ("INT", { "min": 0, "max": MAX_RESOLUTION }),
                "image_height": ("INT", { "min": 0, "max": MAX_RESOLUTION }),
                "points_list": ("STRING", {"multiline": True, "default": ""}),
                "keypoints_type": ("STRING", {"multiline": False, "default": "pose_keypoints_2d"}),
            },
            "optional": {
                "person_number": ("INT", { "default": 0 }),
            }
        }

    CATEGORY = "utils"

    def get_keypoint_from_list(self, list, item):
        idx_x = item*3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return (list[idx_x], list[idx_y], list[idx_conf])

class OpenPoseKeyPointExtractor(OpenPoseKeyPointBase):
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("x", "y", "width", "height")
    FUNCTION = "box_keypoints"

    def box_keypoints(self, pose_keypoint, image_width, image_height, points_list, keypoints_type, person_number=0):
        points_we_want = [int(element) for element in points_list.split(",")]

        min_x = MAX_RESOLUTION
        min_y = MAX_RESOLUTION
        max_x = 0
        max_y = 0
        for element in points_we_want:
            try:
                (x,y,z) = self.get_keypoint_from_list(pose_keypoint[0]["people"][person_number][keypoints_type], element)
            except (IndexError, KeyError):
                print(f"Failed to extract keypoint {element} for person {person_number}")
                continue
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        return (int(min_x*image_width), int(min_y*image_height), int((max_x-min_x)*image_width), int((max_y-min_y)*image_height))

class OpenPoseKeyPointListExtractor(OpenPoseKeyPointBase):
    RETURN_TYPES = ("LIST",)  # Changed return type to LIST
    RETURN_NAMES = ("points",)
    FUNCTION = "box_keypoints"

    def box_keypoints(self, pose_keypoint, image_width, image_height, points_list, keypoints_type, person_number=0):
        points_we_want = [int(element) for element in points_list.split(",")]

        points = []
        for element in points_we_want:
            try:
                (x,y,z) = self.get_keypoint_from_list(pose_keypoint[0]["people"][person_number][keypoints_type], element)
                if x > 0 and y > 0:  # Skip if coordinates are 0 which usually indicates missing keypoint
                    points.append((int(x*image_width), int(y*image_height)))
            except (IndexError, KeyError):
                print(f"Failed to extract keypoint {element} for person {person_number}")
                continue

        print(f"Points extracted: {points}")
        return (points,)  # Return as tuple with single list element

class OpenPoseKeyPointToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("LIST",),  # Changed input type to LIST
                "width": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 64}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "utils"

    def create_mask(self, points, width, height):
        if len(points) <= 2:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
            return (empty_mask,)

        print(f"Creating mask from points: {points}")
        
        # Create numpy mask first using cv2
        points_array = np.array(points, dtype=np.int32)  # points is now already a list
        points_array = points_array.reshape((-1, 1, 2))
        np_mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(np_mask, [points_array], 1.0)
        
        # Convert to tensor with batch dimension
        mask = torch.from_numpy(np_mask[None, ...]).to(dtype=torch.float32, device="cpu")
        
        return (mask,)

class PointListInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("STRING", {"multiline": True, "default": "0,0\n100,100\n200,200"})
            }
        }
    
    RETURN_TYPES = ("LIST",)
    FUNCTION = "parse_points"
    CATEGORY = "utils"

    def parse_points(self, points):
        # Split into lines and filter empty lines
        lines = [line.strip() for line in points.split('\n') if line.strip()]
        point_list = []
        
        for line in lines:
            coords = line.split(',')
            if len(coords) == 2:
                try:
                    x, y = int(coords[0]), int(coords[1])
                    point_list.append((x, y))
                except ValueError:
                    continue
        return (point_list,)

class PointListCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_1": ("LIST",),
                "points_2": ("LIST",)
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "combine_points"
    CATEGORY = "utils"

    def combine_points(self, points_1, points_2):
        combined = list(points_1) + list(points_2)
        return (combined,)

NODE_CLASS_MAPPINGS = {
    "Openpose Keypoint Extractor": OpenPoseKeyPointExtractor,
    "Openpose Keypoint List Extractor": OpenPoseKeyPointListExtractor,
    "Openpose Keypoint To Mask": OpenPoseKeyPointToMask,
    "Point List Input": PointListInput,
    "Point List Combiner": PointListCombiner,
}
