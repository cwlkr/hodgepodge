"""
Author Cédric Walker ITPA Vetsuisse, date 06.09.2021
Utility script for parsing binary masks to shapely objects.
polygon_from_mask parses the full cv2.findContour hierarchy,
making it possiple to parse complcated and casecased polygons.
"""
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from warnings import warn


warning = 'Shapely encountered a linear ring in the data. \
 Some elements in the data where skipped. Set resize=True to prevent this'

def polygon_from_mask(img, resize=False):
    """
    Parsing of binary Mask to shapely MultiyPolygon based on the parsing of the
    cv2.findContours hierarchy.
    Important: Given the shapely assertions single pixel or line elements in
    the Mask can not be converted to shapely.Polygon. To circumvent this
    restriction resize=True resizes the Mask with
    cv2.resize(img, (0,0), fy=2,fy=2, interpolation=cv2.INTER_NEAREST) and then
    scales the polyon back with affinity.scale(pol, 0.5 ,0.5, origin=(0,0)).


    Parameters
    ----------
    img : numpy.ndarray
        binary mask of dimension 2 of type uint8 with positive value of 255
    resize : bool
        resizes the mask and the polygon to circumvent the shapely assertion
        if single pixel or line elements are present in the binary Mask
        default is false.

    Returns
    -------
    shapely.geometry.MultiPolygon
        MultiPolygon of the geometry outlined by the binary mask

    """
    assert(img.ndim == 2)
    assert(img.dtype == np.uint8)
    assert(np.any(img == 255))

    if resize:
        img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    c, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    del(img)  # image might be large and we don't need image information after this point
    all_idx = np.arange(0, len(c))
    layer1_idx = np.argwhere(h[0, :, 3] == -1).squeeze()
    if layer1_idx.shape == ():
        layer1_idx = layer1_idx[None]  # if only one polyon is present in the mask
    pol_list = []
    visited_as_child = []  # are the candiate parents for new layers to check
    layer_idx = layer1_idx
    while len(all_idx) != 0:
        for ind in layer_idx:
            child_idx = np.argwhere(h[0, :, 3] == ind).squeeze()
            if child_idx.shape == ():
                child_idx = child_idx[None]  # if there is a single child for polygon
            inner = [c[i_ind].squeeze() for i_ind in child_idx.tolist()]
            if c[ind].squeeze().shape != (2,):
                try:
                    pol_list.append(Polygon(c[ind].squeeze(), inner))
                except ValueError:
                    warn(warning)
            visited_as_child.extend(child_idx)
            all_idx = np.delete(all_idx, np.where(np.in1d(all_idx, child_idx))[0])
        all_idx = np.delete(all_idx, np.where(np.in1d(all_idx, layer_idx))[0])
        layer_idx = all_idx[np.where(np.in1d(h[0, all_idx, :][:, 3], visited_as_child))[0]]
    pol = unary_union(pol_list)
    if resize:
        pol = affinity.scale(pol, 0.5, 0.5, origin=(0, 0))
    return pol


if __name__ == '__main__':

    # example 1

    import matplotlib.pyplot as plt
    img = cv2.imread('find_contours_example2.png')
    img = img[:, :, 0]
    pol = polygon_from_mask(img, False)
    plt.imshow(img)

    # example 2 with warning and resize=True
    img[1, 2] = 255  # point within image in r,c
    img[1, 1] = 255  # point within image in r,c
    pol = polygon_from_mask(img, False)

    pol = polygon_from_mask(img, True)

    # Properties of polygon

    # find point in inner cutout of polygon to see if a non zero distance is found
    img[230, 200]  # point within image in r,c, within cutout of inner polygon
    from shapely.geometry import Point
    Point(200, 230).distance(pol)  # non zero distance!
    pol.intersects(Point(200, 230))  # point and polygon do not intersect

    # find point in inner polygon to see if zero distance is found
    img[240, 500]  # point within image in r,c, within inner polygon
    Point(500, 240).distance(pol)  # non zero distance!
    pol.intersects(Point(500, 240))  # point and polygon do not intersect
