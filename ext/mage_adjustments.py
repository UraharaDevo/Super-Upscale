import numpy as np
import cv2
import os


class ImageAdjustments:
    def __init__(self, color_level):
        """
        Initializes the ImageAdjustments class with levels settings provided as a dictionary.
        """
        self.shadows = color_level["shadows"] / 255.0
        self.midtones = color_level["midtones"]
        self.highlights = color_level["highlights"] / 255.0
        self.output_shadows = color_level.get("output_shadows", 0) / 255.0
        self.output_highlights = color_level.get("output_highlights", 255) / 255.0

    def color_level(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the level adjustments to an image.
        
        Parameters:
        - image: The input image as a numpy array.
        
        Returns:
        - The adjusted image as a numpy array.
        """
        # Clip the shadows and highlights
        clipped = np.clip((image - self.shadows) / (self.highlights - self.shadows), 0, 1)
        # Apply gamma correction to midtones
        corrected = np.power(clipped, 1 / self.midtones)
        # Scale to the output range
        scaled = corrected * (self.output_highlights - self.output_shadows) + self.output_shadows
        # Ensure the final values are within byte range
        final_image = np.clip(scaled * 255, 0, 255).astype(np.uint8)
        
        return final_image
    
    def batch_mode(self, input_path, output_path):
        """
        Applies the level adjustments to all images in the input directory and saves
        them to the output directory.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    result_img = self.color_level(img)
                    cv2.imwrite(os.path.join(output_path, filename), result_img)
                else:
                    print(f"Skipping file {filename}, not a valid image.")