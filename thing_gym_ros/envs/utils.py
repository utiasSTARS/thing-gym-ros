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

def crop_img(img, relative_corners):
    """ relative_corners are floats between 0 and 1 designating where the corners of a crop box
    should be ([[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]]).

    e.g. [[0, 0], [1, 1]] would be the full image, [[0.5, 0.5], [1, 1]] would be bottom right."""
    rc = relative_corners
    raw_height, raw_width = img.shape[:2]
    top_left_pix = [int(rc[0][0] * raw_width), int(rc[0][1] * raw_height)]
    bottom_right_pix = [int(rc[1][0] * raw_width), int(rc[1][1] * raw_height)]
    img_cropped = img[top_left_pix[1]:bottom_right_pix[1], top_left_pix[0]:bottom_right_pix[0]]
    return img_cropped