import rclpy
import rclpy.logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

import gc
from model_args import aot_args,sam_args,segtracker_args
from SegTracker import SegTracker
from seg_track_anything import draw_mask
# import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/color/resize/image_raw',
            self.listener_callback,
            10)

        self.mask_pub = self.create_publisher(Image, 'mask', 10)
        self.masked_frame_pub = self.create_publisher(Image, 'masked_frame', 10)

        self.cv_bridge = CvBridge()

        sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 100,
        }
            
        segtracker_args = {
            'sam_gap': 10, # the interval to run sam to segment new objects
            'min_area': 200, # minimal mask area to add a new mask as a new object
            'max_obj_num': 30, # maximal object number to track in a video
            'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
        }

        self.frame_idx = 0

        rclpy.logging.get_logger('rclpy').info('Loading SAM Tracker model...')
        torch.cuda.empty_cache()
        gc.collect()
        self.sam_gap = segtracker_args['sam_gap']
        self.segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.segtracker.restart_tracker()

        # rclpy.logging.get_logger('rclpy').info('SAM model loading...')
        # sam = sam_model_registry["vit_h"](checkpoint="/home/dlr-rmc/hjkim/segment_ws/checkpoint/sam_vit_h_4b8939.pth")
        # sam.to(device = "cuda")
        # self.predictor = SamPredictor(sam)

        self.points = np.array([[320, 180]])
        self.prompt = {
            "points_coords" : self.points,
            "points_modes" : np.array([1]),
            "multimask" : False,
        }

        self.seg_all = False
        self.pred_mask = None
        self.obj_ids = 0

    def listener_callback(self, msg):
        stamp = msg.header.stamp
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')
        pred_mask, masked_frame = self.sam_tracking(cv_image)
        self.pred_mask = pred_mask
        self.pub_mask(pred_mask, stamp)
        self.pub_masked_frame(masked_frame, stamp)

    def sam_tracking(self, cv_image):
        with torch.cuda.amp.autocast():
            frame = cv_image
            
            if self.frame_idx == 0:
                point_id = 0
            else:
                for i in range(-2,3):
                    for j in range(-2,3):
                        point_id = self.pred_mask[self.points[0][1]-i, self.points[0][0]-j]
                        if point_id != 0:
                            break
                    if point_id != 0:
                        break

            if point_id == 0:
                rclpy.logging.get_logger('rclpy').info('Adding reference')
                # self.segtracker.restart_tracker()

                self.segtracker.sam.interactive_predictor.set_image(frame)
                pred_mask = self.segtracker.sam.segment_with_click(
                                            origin_frame=frame,
                                            coords=self.prompt["points_coords"],
                                            modes=self.prompt["points_modes"],
                                            multimask=self.prompt["multimask"])
                if self.frame_idx == 0:
                    self.segtracker.add_reference(frame, pred_mask)
                    self.obj_ids += 1

                else:
                    track_mask = self.segtracker.track(frame, update_memory=True)
                    rclpy.logging.get_logger('rclpy').info('track ids: {}'.format(np.unique(track_mask)))
                    seg_mask = np.where(pred_mask != 0, pred_mask + self.obj_ids, pred_mask)
                    rclpy.logging.get_logger('rclpy').info('seg ids: {}'.format(np.unique(seg_mask)))

                    new_obj_mask = self.segtracker.find_new_objs(track_mask,seg_mask)
                    

                    if np.max(new_obj_mask) != 0:
                        new_obj_mask = np.where(new_obj_mask != 0, new_obj_mask + 1, new_obj_mask)
                        pred_mask = track_mask + new_obj_mask
                        self.segtracker.add_reference(frame, pred_mask)
                        self.obj_ids += 1
                    else:
                        pred_mask = track_mask    
                
            else:
                pred_mask = self.segtracker.track(frame, update_memory=True)
            
            torch.cuda.empty_cache()
            gc.collect()

            class_ids = np.unique(pred_mask)
            rclpy.logging.get_logger('rclpy').info('class ids: {}'.format(class_ids))
            
            masked_frame = draw_mask(frame, pred_mask)
            self.frame_idx += 1
        return pred_mask, masked_frame
            
    def pub_mask(self, mask, stamp):
        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header.stamp = stamp

        self.mask_pub.publish(mask_msg)
    
    def pub_masked_frame(self, masked_frame, stamp):
        masked_frame = cv2.circle(masked_frame, tuple(self.points[0]), 5, (0, 0, 255), -1)
        masked_frame_msg = self.cv_bridge.cv2_to_imgmsg(masked_frame, encoding="rgb8")
        masked_frame_msg.header.stamp = stamp
        self.masked_frame_pub.publish(masked_frame_msg)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
