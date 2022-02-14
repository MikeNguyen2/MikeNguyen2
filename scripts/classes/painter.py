"""This document decribes a painter."""
import cv2 as cv


class Painter:
    """Create a painter, which can draw items on images."""

    def __init__(self):
        """Initialize the painter."""
        pass

    def paint_microplate(self, image, microplate):
        """Draw a microplate on an image and return it."""
        for row in microplate.wells2d:
            for well in row:
                #print(well.position)
                #print(dir(well))
                position = (well.position[0], well.position[1])
                image = cv.circle(
                    image, position, 8, (255, 255, 0), 1, cv.LINE_AA
                )

        return image

    def paint_tube(self, image):
        """Draw a tube on an image and return it."""
        # TODO paint the tube into the image
        return image


if __name__ == '__main__':
    painter = Painter()
