import cv2
import numpy as np 
import pyrealsense2 as rs

class depthMap:

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60) #360
        self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
        self.align = rs.align(rs.stream.color)

        self.profile = self.pipeline.start(self.config)
        
    def _get_depth_map(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        color_frame_np = np.array(color_frame.get_data())
        depth_frame_np = np.array(depth_frame.get_data())
    
        return color_frame_np, depth_frame_np

    def _debug_show_frames(self, color_frame_np, depth_frame_np):   
        depth_frame_16 = depth_frame_np
        df_dp = np.expand_dims(depth_frame_16, axis=-1).astype(np.uint8)
        df_dp = np.tile(df_dp, (1, 1, 3))
        depth_frame = depth_frame_16.astype(np.float32)
        cv2.imshow('color_debug',color_frame_np)
        cv2.imshow('depth_debug',depth_frame_np)

if __name__ == '__main__':
    cam = depthMap()

    while(True):
        color_frame_np, depth_frame_np = cam._get_depth_map()
        cam._debug_show_frames(color_frame_np, depth_frame_np)
        if(cv2.waitKey(1) == 27):
            break
        

    cv2.destroyAllWindows()


