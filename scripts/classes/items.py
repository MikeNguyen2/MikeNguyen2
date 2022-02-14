"""This document contains interactive objects."""


class Item:
    """The standardclass for the classes below."""

    def __init__(self, position, name):
        """
        Initialize the core parameters.

        position:   tuple,      the position of the Item
        name:       String,     the name of the Item
        """
        self.position = position
        self.name = name


class Item2D(Item):
    """The standardclass for 2d-items."""

    def __init__(self, position, contour, name):
        """
        Initialize the core parameters.

        position:   tuple,      the position of the Item
        name:       String,     the name of the Item
        contour:    array,      the contour of the Item
        """
        Item.__init__(self, position, name)
        self.contour = contour


class Item3D(Item):
    """The standardclass for 3d-items."""

    def __init__(self, position, name):
        """
        Initialize the core parameters.

        position:   tuple,      the position of the Item
        name:       String,     the name of the Item
        """
        Item.__init__(self, position, name)


class Microplate2D(Item2D):
    """Create an image-Microplate."""

    def __init__(self, position, contour, wells2d):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        contour:    array,      the contour of the Item
        wells2d     array,      objects of the type 'Well2D'
        """
        Item2D.__init__(
            self, position, contour, 'Microplate2D'
        )
        self.wells2d = wells2d


class Microplate3D(Item3D):
    """Create a real world Microplate."""

    def __init__(self, position, wells3d):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        wells3d     array,      objects of the type 'Well3D'
        """
        Item3D.__init__(
            self, position, 'Microplate3D'
        )
        self.wells3d = wells3d


class Well2D(Item2D):
    """Create an image-Well."""

    def __init__(self, contour, position, form='circle'):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        contour:    array,      the contour of the Item
        form:       String,     the shape of the Item
        """
        Item2D.__init__(
            self, contour, position, 'Well2D'
        )
        self.form = form


class Well3D(Item3D):
    """Create an real world Well."""

    def __init__(self, position):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        """
        Item3D.__init__(
            self, position, 'Well3D'
        )


class Tube2D(Item2D):
    """Create an image-tube."""

    def __init__(self, contour, position, qr_code):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        contour:    array,      the contour of the Item
        qr_code:    array,      the qr_code of the Item
        """
        Item2D.__init__(
            self, contour, position, 'Tube2D'
        )
        self.qr_code = qr_code  # TODO Add qr code reading


class Tube3D(Item3D):
    """Create an real world tube."""

    def __init__(self, position):
        """Initialize the Parameters.

        position:   tuple,      the position of the Item
        """
        Item3D.__init__(
            self, position, 'Tube2D'
        )  # TODO Add volume, width, height and depth


if __name__ == '__main__':
    well = Well2D(None, None, None)
