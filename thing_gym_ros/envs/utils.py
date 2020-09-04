""" Various generic env utilties. """


def center_crop_img(img, crop_zoom):
    """ crop_zoom is amount to "zoom" into the image. E.g. 2.0 would cut out half of the width,
    half of the height, and only give the center. """
    raw_height, raw_width = img.shape[:2]
    center = raw_height // 2, raw_width // 2
    crop_size = raw_height // crop_zoom, raw_width // crop_zoom
    min_y, max_y = int(center[0] - crop_size[0] // 2), int(center[0] + crop_size[0] // 2)
    min_x, max_x = int(center[1] - crop_size[1] // 2), int(center[1] + crop_size[1] // 2)
    img_cropped = img[min_y:max_y, min_x:max_x]
    return img_cropped