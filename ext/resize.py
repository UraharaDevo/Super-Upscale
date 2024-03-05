from chainner_ext import resize, ResizeFilter
import numpy as np
import os
import cv2

class Resize:
    def __init__(self, setting):
        self.size = setting["size"]
        self.width = setting["width"]
        self.interpolation = setting["interpolation"]
        self.percent = setting["percent"] / 100
        self.spread = setting["spread"]
        self.spread_size = setting["spread_size"]
        self.interpolation_map = self._create_interpolation_map(setting["interpolation"])

    def _create_interpolation_map(self, interpolation):
        if interpolation == "auto":
            # Logic to choose interpolation based on upsizing or downsizing
            return {
                'upscale': ResizeFilter.Cubic,
                'downscale': ResizeFilter.Lanczos
            }
        else:
            return {
                interpolation: ResizeFilter[interpolation]
            }

    def _determine_interpolation(self, original_size, new_size):
        if new_size > original_size:
            return self.interpolation_map.get('upscale', ResizeFilter.Cubic)
        else:
            return self.interpolation_map.get('downscale', ResizeFilter.Lanczos)
        
        
    def Factor(self, img):
        """تطبيق النسبة المئوية لتغيير حجم الصورة."""
        height, width = img.shape[:2]
        new_width = int(width * self.percent / 100)
        new_height = int(height * self.percent / 100)
        return (new_width, new_height)
    
    def resize_to_side(self, img):
        """تغيير حجم الصورة إلى جانب محدد باستخدام spread."""
        height, width = img.shape[:2]
        if self.width:
            new_height = self.spread_size
            new_width = int(width / height * self.spread_size)
        else:
            new_width = self.spread_size
            new_height = int(height / width * self.spread_size)
        return (new_width, new_height)
    
    def batch_mode(self, input_path, output_path):
        """تغيير حجم جميع الصور في مجلد وحفظ النتائج في مجلد آخر باستخدام OpenCV."""
        # التأكد من وجود المجلد الناتج
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # قراءة جميع الملفات في المجلد المدخل
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)  # قراءة الصورة باستخدام OpenCV
                
                if img is not None:
                    # تطبيق تغيير الحجم
                    resized_img = self.run(img)
                    
                    # حفظ الصورة المعدلة باستخدام OpenCV
                    cv2.imwrite(os.path.join(output_path, filename), resized_img)
                else:
                    print(f"Skipping file {filename}, not a valid image or cannot be read.")
        

    def run(self, img):
        original_height, original_width = img.shape[:2]
        if self.spread and ((self.width and original_width > self.spread_size) or (not self.width and original_height > self.spread_size)):
            new_size = self.resize_to_side(img)
        else:
            new_size = self.Factor(img)
        interpolation_method = self._determine_interpolation(original_width if self.width else original_height, new_size[0] if self.width else new_size[1])
        resized_img = resize(img.astype(np.float32), new_size, interpolation_method, gamma_correction=False)
        return resized_img
