#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
import matplotlib.pyplot as plt
import math

import scipy
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d, medfilt

from skimage.morphology import skeletonize
from skimage.filters import sobel
from skimage import feature

#################################
#
# Imports from CellProfiler
#
##################################
from cellprofiler_core.module import Module
from cellprofiler_core.setting.subscriber import LabelSubscriber
import pickle
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT

__doc__ = """
MeasureMinMaxWidth
===================
**MeasureMinMaxWidth** - Measures the minimum and maximum width of the given 
objects in an image.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

Input to the module should be objects you want to measure.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

The output of this module is the maximum and minimum width of the given object. 

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


**Measurements:**

-  *MaxWidth*: A brief description of the measurement.
-  *MinWidth*: A brief description of the measurement.
-  *AverageWidth*: .

Technical notes
^^^^^^^^^^^^^^^

This module works by skeletonizing the given object before measuring the minimum 
distance between each pixel of the skeleton and the closest edge of the mask.

"""

def find_endpoints(img):
    # Kernel to sum the neighbours
    kernel = [[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]]
   
    # 2D convolution (cast image to int32 to avoid overflow)
    img_conv = convolve2d(img.astype(np.int32), kernel, mode='same')
    
    # Pick points where pixel is 255 and neighbours sum 255
    endpoints = np.stack(np.where((img == 255) & (img_conv == 255)), axis=1)
    
    return endpoints


def extend_skeleton(skeleton, outline, sobel, line_end):
    
    local_skeleton = skeleton.copy()
    end_x = line_end[1]
    end_y = line_end[0]
                    
    while True:
        end_pixel = outline[end_y, end_x]
        
        # If the end pixel is on the outline the skeleton extension process is finished
        if end_pixel == 255:
            break
            
        # If the end pixel is not on the outline we need to find the nearest pixel (out of the neighbouring pixels that are the best to extend along)
        # Note we need to exclude the exisiting skeleton from this
        
        # Get sobel neighbours and skel neigbours
        sobel_neighbours = sobel[end_y-1:end_y+2, end_x-1:end_x+2]
        skeleton_neighbours = np.invert(local_skeleton[end_y-1:end_y+2, end_x-1:end_x+2])
        
        # Filter out skel neighbours if present
        
        possible_routes = np.divide(skeleton_neighbours * sobel_neighbours, 255)
        
        if possible_routes.size == 0:
            break

        possible_routes[possible_routes==0] = 'nan'
        
        # Select best sobel neighbour
        rel_cords = np.where(possible_routes == np.nanmin(possible_routes))
        
        if len(rel_cords[0]) == 0:
            break
        
        
        coord_translation = [rel_cords[0][0] -1, rel_cords[1][0] -1]
        
        # New end of line coord
        end_y = end_y + coord_translation[0]
        end_x = end_x + coord_translation[1]
        
        #Update local skel
        local_skeleton[end_y, end_x] = 255  
    
    return local_skeleton


def process_object(image):

    # Smoothed image
    #smoothed_image = medfilt(image, kernel_size=25)

    
    # Skeletonize image
    skeleton = skeletonize(image)
    skeleton_im = skeleton * np.uint8(255)

    # Get outline + smooth image
    #outline = cv2.Canny(image, 0,0)
    outline = feature.canny(image.astype('float')) * np.uint8(255)

    edt = distance_transform_edt(image) 
    sobel_edt = sobel(edt)

    # Work out inital end of skeleton
    end_points = find_endpoints(skeleton_im)

    extended_skeleton = skeleton_im.copy()

    for end_point in end_points:
        extended_skeleton = extend_skeleton(extended_skeleton, outline,sobel_edt, end_point)

    widths = distance_transform_edt(image)
    skeleton_only_width = np.multiply(np.divide(widths * extended_skeleton, 255), 2)    
    skeleton_only_width[skeleton_only_width<14] = 'nan'
    
    skel_length = np.multiply(np.divide(widths * extended_skeleton, 255), 2)
    skel_length[skel_length==0] = 'nan'
    skel_length = len(skel_length[~np.isnan(skel_length)])
    
    adjusted_measurements = skeleton_only_width[~np.isnan(skeleton_only_width)]
    
    try:
        min_width = np.min(adjusted_measurements)
        max_width = np.max(adjusted_measurements)
    except:
        min_width = np.nan
        max_width = np.nan
        
    return min_width, max_width, skeleton, extended_skeleton, outline



class MeasureMinMaxWidth(Module):
   
    module_name = "MeasureMinMaxWidth"
    category = "Measurement"
    variable_revision_number = 1


    def create_settings(self):
        self.input_object_name = LabelSubscriber(
                text="Input object name",
                doc="These are the objects that the module operates on.",
        )
    
    def settings(self):
        return [self.input_object_name]

    def visible_settings(self):
        return [self.input_object_name]

    def run(self, workspace):

        measurements = workspace.measurements

        statistics = [["Min Width", "Max Width"]]
        workspace.display_data.statistics = statistics

        input_object_name = self.input_object_name.value
        object_set = workspace.object_set


        objects = object_set.get_objects(input_object_name)
        labels = objects.segmented

        minws = []
        maxws = []

        for obj_no in objects.indices:
            minw, maxw = self.run_object(obj_no, labels)

            minws.append(minw)
            maxws.append(maxw)
        
        measurements.add_measurement(input_object_name, "Min Width", minws)
        measurements.add_measurement(input_object_name, "Max Width", maxws)
          
        statistics.append([minws, maxws])

 
    def run_object(self, obj_no, full_image):
        obj_image = np.uint8((full_image==obj_no).astype(int))
        minw, maxw, skel, extskel, outline = process_object(obj_image)

        return float(minw), float(maxw)
    
    def get_measurement_columns(self, pipeline):
      
        input_object_name = self.input_object_name.value

        return [
            (input_object_name, "Max Width", COLTYPE_FLOAT),
            (input_object_name, "Min Width", COLTYPE_FLOAT)
        ]


    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics

        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics, col_labels=["Min Width", "Max Width"])

        

