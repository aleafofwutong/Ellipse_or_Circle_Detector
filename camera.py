import pyrealsense2 as rs
import numpy as np
import cv2
class RealSenseCamera:
    def __init__(self, enable_blur=False,blur_kernel_size=7):
        self.enable_blur = enable_blur
        self.blur_kernel_size = blur_kernel_size
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Create an align object
        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if self.enable_blur:
            color_image = cv2.GaussianBlur(color_image, (self.blur_kernel_size, self.blur_kernel_size), 0)

        return depth_image, color_image

    def release(self):
        # Stop streaming
        self.pipeline.stop()
if __name__ == "__main__":
    camera = RealSenseCamera(enable_blur=True)
    try:
        while True:
            depth_image, color_image = camera.get_frames()
            if depth_image is None or color_image is None:
                continue

            # Apply colormap on depth image (for visualization)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.imshow('RealSense', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()