from ext.resize import ImageResizer
settings = {
    'mode': 'factor',
    'side': 'width',
    'spread_size': 700,
    'interpolation': 'auto',
    'scale_factor' : 80
}

resizer = ImageResizer(settings)

input_folder = R"C:\Users\KSAAp\Downloads\rr"
output_folder = R"C:\Users\KSAAp\Downloads\rrs"

resizer.batch_mode(input_folder, output_folder)