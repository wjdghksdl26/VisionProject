import cv2


def SubtractImages(image1, image2, clip=0, absolute=False, isColor=False):
    image1 = image1.astype('int32')
    image2 = image2.astype('int32')

    if absolute is True:
        subt_image = abs(image1 - image2)

    if absolute is False:
        subt_image = image1 - image2

    subt_image = subt_image.clip(clip).astype('uint8')

    return subt_image
