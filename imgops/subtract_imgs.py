import cv2


def subtract_images(image1, image2, clip=0, absolute=False, isColor=False):
    if isColor is True:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype('int32')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype('int32')

    if isColor is False:
        image1 = image1.astype('int32')
        image2 = image2.astype('int32')

    if absolute is True:
        subt_image = abs(image1 - image2)

    if absolute is False:
        subt_image = image1 - image2

    subt_image = subt_image.clip(clip).astype('uint8')

    return subt_image
