import os
from wand.image import Image

class ImageResizer:

    def __init__(self, settings):
        self.settings = settings

    def select_interpolation_method(self, width, height, new_width, new_height):
        if self.settings.get('interpolation') == 'auto':
            if new_width > width or new_height > height:
                return 'lanczos'
            else:
                return 'mitchell'
        else:
            return self.settings.get('interpolation', 'undefined').lower()

    def resize_image(self, input_path, output_path):
        with Image(filename=input_path) as img:
            mode = self.settings.get('mode', 'factor')
            width, height = img.width, img.height
            
            if mode == 'factor':
                scale_factor = self.settings.get('scale_factor', 1)
                new_width = int(width * scale_factor / 100)
                new_height = int(height * scale_factor / 100)
            elif mode == 'side':
                if self.settings.get('side') == 'width':
                    new_width = self.settings.get('spread_size')
                    new_height = int(new_width * (height / width))
                elif self.settings.get('side') == 'height':
                    new_height = self.settings.get('spread_size')
                    new_width = int(new_height * (width / height))
                    
            filter_type = self.select_interpolation_method(width, height, new_width, new_height)
            
            img.resize(new_width, new_height, filter=filter_type)
            img.save(filename=output_path)

    def batch_mode(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                self.resize_image(input_path, output_path)