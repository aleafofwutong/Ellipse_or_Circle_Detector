from camera import RealSenseCamera
import cv2
import numpy as np
print("Imported RealSenseCamera from camera module successfully.")

class ColorFilter:
    def __init__(self, enable_blur=False):
        self.camera = RealSenseCamera(enable_blur=enable_blur)

    def test_run(self):
        try:
            while True:
                depth_image, color_image = self.camera.get_frames()
                if depth_image is None or color_image is None:
                    continue

                # Get color masks
                hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                color_results = self.get_images(hsv_image, color_image)
                
                # Display original image
                cv2.imshow('Original', color_image)
                
                # # Display each color mask
                # cv2.imshow('Red Mask', color_results['red'])
                # cv2.imshow('Blue Mask', color_results['blue'])
                # cv2.imshow('White Mask', color_results['white'])
                # cv2.imshow('Black Mask', color_results['black'])
                cv2.imshow('All Colors', color_results['all'])
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
    def get_images(self, hsv_img, original_img):
        # 红色（两段合并）
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([179, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv_img, lower_red1, upper_red1),
                                cv2.inRange(hsv_img, lower_red2, upper_red2))
        
        # 蓝色
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
        
        # 白色 - 高亮度、低饱和度
        lower_white = np.array([0, 0, 90])
        upper_white = np.array([180, 80, 255])
        mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
        
        # 黑色 - 低亮度
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv_img, lower_black, upper_black)

        # Apply masks to original image
        result_red = cv2.bitwise_and(original_img, original_img, mask=mask_red)
        result_blue = cv2.bitwise_and(original_img, original_img, mask=mask_blue)
        # result_white = cv2.bitwise_and(original_img, original_img, mask=mask_white)
        # result_black = cv2.bitwise_and(original_img, original_img, mask=mask_black)
        
        # Create colored overlay for all colors to show them distinctly
        result_all = np.zeros_like(original_img)
        
        # Add each color region to the combined result
        result_all = cv2.add(result_all, result_red)
        result_all = cv2.add(result_all, result_blue)
        
        # For white regions, ensure they show as bright white
        white_overlay = np.zeros_like(original_img)
        white_overlay[mask_white > 0] = [255, 255, 255]  # Pure white
        result_all = cv2.add(result_all, white_overlay)
        
        # For black regions, show them as dark gray to distinguish from background
        black_overlay = np.zeros_like(original_img)
        black_overlay[mask_black > 0] = [60, 60, 60]  # Dark gray
        result_all = cv2.add(result_all, black_overlay)
        
        return {
            # 'red': result_red,
            # 'blue': result_blue,
            # 'white': result_white,
            # 'black': result_black,
            'all': result_all
        }
    



class Detector:
    def __init__(self):
        self.set={}  
        self.image_flow=ColorFilter(enable_blur=True)

    def detect(self, img):
        # Convert to grayscale if image is colored
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply morphological operations to improve contour detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adjust Canny thresholds for better edge detection
        edges = cv2.Canny(blur, 45, 100)
        
        # Apply morphological closing to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create colored output image for visualization
        output = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        print(f"Found {len(contours)} contours")
        
        detected_count = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            if peri == 0:
                continue
            roundness = (4 * np.pi * area) / (peri ** 2)
            if len(contour) < 20 or roundness < 0.3:
                continue
            
            # Draw all contours for debugging (in blue)
            cv2.drawContours(output, [contour], -1, (255, 0, 0), 1)
            
            try:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                
                # Relaxed size constraints
                if 10 < MA < 300 and 10 < ma < 300 and area > 4000:
                    self.set[(int(x), int(y))] = (int(MA), int(ma), int(angle))
                    cv2.ellipse(output, ellipse, (0, 255, 0), 2)
                    detected_count += 1
                    print(f"Detected ellipse #{detected_count} at ({int(x)}, {int(y)}) with axes ({int(MA)}, {int(ma)}), angle {int(angle)}, area {int(area)}")
            except:
                pass
        
        print(f"Total ellipses detected: {detected_count}")
        
        # Show edges for debugging
        cv2.imshow('Edges', edges)
        
        return output
    def run(self):
        while True:
            depth_image, color_image = self.image_flow.camera.get_frames()
            if depth_image is None or color_image is None:
                continue
            img=self.image_flow.get_images(cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV), color_image)['all']
            
            # Show filtered image for debugging
            cv2.imshow('Filtered', img)
            
            detected_image = self.detect(img)

            cv2.imshow('Detected', detected_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
if __name__ == "__main__":
    # Test color filter only
    # color_filter = ColorFilter(enable_blur=True)
    # color_filter.test_run()
    
    # Run detector with ellipse detection
    detector = Detector()
    detector.run()