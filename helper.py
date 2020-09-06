import imutils as im
import math
# generator function that returns multiple windows
# params: 
# image, ws: windows size(length and breadth)
# step: number of pixels to skip for sliding a windows
def sliding_window(image, step, ws):
    # start sliding window
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] -ws[0], step):
            # returns a window of image
            yield(x,y,image[y:y+ws[1], x:x+ws[0]])


# generator function that returns image pyramids
# params:
# image, scale: by which we reduce image width
# minsize: minsize after which we stop creating image pyramids
def image_pyramid(image, scale=1.5, minsize=(224,224)):
    # currently returns original image
    yield image

    while True:
        # calculates new width of the image
        newWidth = int(image.shape[1] / scale)
        # resizing the image with new width
        image = im.resize(image, width=newWidth)
        # check if image size is less than the min size
        if image.shape[1] < minsize[0] and image.shape[0] < minsize[1]:
            break
        # if not we return the image pyramid
        yield image