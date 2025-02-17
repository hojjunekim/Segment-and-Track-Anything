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

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',
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
            'max_obj_num': 50, # maximal object number to track in a video
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

        self.points = np.array([[740, 360]])
        self.prompt = {
            "points_coord" : self.points,
            "points_mode" : np.array([1]),
            "multimask" : "False",
        }

        self.seg_all = False
        self.pred_mask = None

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
            
            # if self.frame_idx == 0:
            pred_mask, annotated_frame = self.segtracker.detect_and_seg(
                                            origin_frame = frame,
                                            grounding_caption = "blocks with an ar tag",
                                            box_threshold = 0.5,
                                            text_threshold = 0.3,
                                            box_size_threshold = 0.3)
                
            torch.cuda.empty_cache()
            gc.collect()

            class_ids = np.unique(pred_mask)
            class_ids = class_ids[class_ids!=0]
            rclpy.logging.get_logger('rclpy').info('class ids: {}'.format(class_ids))
            
            masked_frame = draw_mask(annotated_frame, pred_mask, id_countour=False)
            self.frame_idx += 1
        return pred_mask, masked_frame
            
            # if self.frame_idx == 0:
            #     rclpy.logging.get_logger('rclpy').info('First frame')
                
            #     if self.seg_all:
            #         pred_mask = self.segtracker.seg(frame)
            #     else:
            #         self.segtracker.sam.interactive_predictor.set_image(frame)
            #         pred_mask = self.segtracker.sam.segment_with_click(
            #                                   origin_frame=frame,
            #                                   coords=self.prompt["points_coord"],
            #                                   modes=self.prompt["points_mode"],
            #                                   multimask=self.prompt["multimask"])
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     # self.segtracker.add_reference(frame, pred_mask, self.frame_idx)
            #     # self.segtracker.first_frame_mask = pred_mask

            # elif (self.frame_idx % self.sam_gap) == 0:
            #     rclpy.logging.get_logger('rclpy').info('Segmenting')
            #     if self.seg_all:
            #         seg_mask = self.segtracker.seg(frame)
            #     else:
            #         seg_mask = self.segtracker.sam.segment_with_click(
            #                                   origin_frame=frame,
            #                                   coords=self.prompt["points_coord"],
            #                                   modes=self.prompt["points_mode"],
            #                                   multimask=self.prompt["multimask"])
            #     # seg_mask = self.segtracker.seg(frame)
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     # track_mask = self.segtracker.track(frame)
            #     pred_mask = seg_mask
            #     # find new objects, and update tracker with new objects
            #     # new_obj_mask = self.segtracker.find_new_objs(track_mask,seg_mask)
            #     # pred_mask = track_mask + new_obj_mask
            #     # self.segtracker.restart_tracker()
            #     # self.segtracker.add_reference(frame, pred_mask, self.frame_idx)
            # else:
            #     rclpy.logging.get_logger('rclpy').info('Tracking')
            #     seg_mask, _ = self.segtracker.seg_acc_click(
            #                                   origin_frame=frame,
            #                                   coords=self.prompt["points_coord"],
            #                                   modes=self.prompt["points_mode"],
            #                                   multimask=self.prompt["multimask"])
            #     # pred_mask = self.segtracker.track(frame,update_memory=True)
            #     pred_mask = seg_mask
            # torch.cuda.empty_cache()
            # gc.collect()

            
    # def sam(self, cv_image):
    #     self.predictor.set_image(cv_image)
    #     size = cv_image.shape[:2]
    #     input_point = np.array([[size[1]//2, size[0]//2]])
    #     input_label = np.array([1])

    #     masks, scores, logits = self.predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         multimask_output=False,
    #     )
    #     rclpy.logging.get_logger('rclpy').info('Predicted masks # {}'.format(len(masks)))

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
