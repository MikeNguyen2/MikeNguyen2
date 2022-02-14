"""This document describes an image classifier."""
import numpy as np
import cv2 as cv
import math
import random
import items


class Classifier:
    """Create a classifier to classify objects in an image."""

    def __init__(self):
        """Initialize the classifier."""
        pass

    def __is_close(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def __auto_canny(self, image, sigma):
        median = np.median(image)
        lower_limit = int(max(0, (1 - sigma) * median))
        upper_limit = int(min(255, (1 + sigma), median))
        return cv.Canny(image, lower_limit, upper_limit)

    def __mapping(self, value, a, b, c, d):
        return c + (d - c) * ((value - a) / float(b - a))

    def __calculate_pattern(self, points, rows, columns, margin1, margin2):
        shortest_distance = 10**10
        closest_point = None
        random_point = random.choice(points)
        for point in points:
            if np.array_equal(random_point, point):
                continue

            print(random_point)
            print(point)
            distance = np.linalg.norm(random_point-point)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_point = point

        pair1 = [random_point, closest_point]
        pair2 = []
        for point in points:
            if np.array_equal(pair1[0], point):
                continue
            if np.array_equal(pair1[1], point):
                continue

            pair2.append(point)

        pair1_average_x = (pair1[0][0] + pair1[1][0]) / 2.0
        pair2_average_x = (pair2[0][0] + pair2[1][0]) / 2.0

        top_left = None
        if pair1_average_x < pair2_average_x:
            if pair1[0][1] < pair1[1][1]:
                top_left = pair1[0]
            else:
                top_left = pair1[1]
        else:
            if pair2[0][1] < pair2[1][1]:
                top_left = pair2[0]
            else:
                top_left = pair2[1]

        shortest_distance = 10**10
        closest = None
        longest_distance = 0
        furthest = None
        for point in points:
            if np.array_equal(top_left, point):
                continue

            distance = np.linalg.norm(top_left-point)
            if distance < shortest_distance:
                shortest_distance = distance
                closest = point

            if distance > longest_distance:
                longest_distance = distance
                furthest = point

        closest2 = None
        for point in points:
            if np.array_equal(top_left, point):
                continue
            if np.array_equal(furthest, point):
                continue
            if np.array_equal(closest, point):
                continue
            closest2 = point

        con_closest = [closest[0]-top_left[0], closest[1]-top_left[1]]
        con_closest2 = [closest2[0]-top_left[0], closest2[1]-top_left[1]]

        um = 1-margin1
        lm = margin1
        um2 = 1-margin2
        lm2 = margin2

        positions = []
        for j in range(12):
            positions.append([])
            for i in range(8):
                well_x = top_left[0]
                well_x += self.__mapping(
                    i, 0, rows-1, con_closest[0]*lm2, con_closest[0]*um2)
                well_x += self.__mapping(
                    j, 0, columns-1, con_closest2[0]*lm, con_closest2[0]*um)

                well_y = top_left[1]
                well_y += self.__mapping(
                    i, 0, rows-1, con_closest[1]*lm2, con_closest[1]*um2)
                well_y += self.__mapping(
                    j, 0, columns-1, con_closest2[1]*lm, con_closest2[1]*um)

                positions[j].append([np.float32(well_x), np.float32(well_y)])

        return positions

    def find_microplate(self, image):
        """
        Return a microplate object when found, else return NULL.

        image:      array,              the image to classify
        return:     microplate or NULL, the classified object
        """
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_blur = cv.blur(image_gray, (3, 3))
        image_canny = self.__auto_canny(image_blur, 0.33)  # TODO Try sigmas
        image_adaptive = cv.adaptiveThreshold(image_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3)
        cv.imshow('image', image)
        cv.imshow('gray', image_gray)
        cv.imshow('blur', image_blur)
        cv.imshow('canny', image_canny)
        cv.imshow('adaptive', image_adaptive)
        # cv.waitKey(0)
        contours, _ = cv.findContours(
            image_adaptive, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
        )
        for contour in contours:
            sigma = 0.12  # TODO Try sigmas
            length = cv.arcLength(contour, False)
            approx = cv.approxPolyDP(contour, sigma*length, False)
            number_of_corners = len(approx)
            if number_of_corners < 4 or number_of_corners > 5:
                # TODO Maybe allow 5 corners aswell
                continue
            print(number_of_corners)

            minAreaRect = cv.minAreaRect(contour)
            center, (length1, length2), angle = minAreaRect
            # TODO Maybe use 4 points directly
            # TODO Calculate angle from 90 to 180

            tolerance = 0.3  # TODO Try out different values

            long = max(length1, length2)
            print(long)
            long_expected = 270  # pixels
            if not self.__is_close(long, long_expected, rel_tol=tolerance):
                continue
            print('found right length for long')

            short = min(length1, length2)
            print(short)
            short_expected = 180  # pixels
            if not self.__is_close(short, short_expected, rel_tol=tolerance):
                continue
            print('found right length for short')

            corner_points = cv.boxPoints(minAreaRect)
            corner_points_np = []
            for corner_point in corner_points:
                corner_points_np.append(np.array(corner_point))

            well_points = self.__calculate_pattern(
                corner_points_np, 8, 12, 0.11, 0.14
            )

            wells = []
            for row in well_points:
                column = []
                for well_point in row:
                    # TODO Add contour and form detection
                    column.append(items.Well2D(well_point, None))
                wells.append(column)

            return items.Microplate2D(center, contour, wells)

    def find_tube(self, image):
        """
        Return a tube object when found, else return NULL.

        image:      array,              the image to classify
        return:     tube or Null,       the classified object
        """

        pass


if __name__ == '__main__':
    classifier = Classifier()
