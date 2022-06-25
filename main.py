import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from scipy import ndimage as ndi
import dash_components.slice_mapper_util
import dash_components.slice_mapper as slice_mapper
from dash_components.slice_mapper import SliceMapper
from scipy.ndimage import map_coordinates
from PIL import Image
# This is a sample Python script.

import dash_components.slice_mapper_util as smutil

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

cam = [[12.38505809, 5.64245275],
 [12.17361543, 10.92851918],
 [11.11640214, 14.31160169],
 [12.59650074, 18.32901218],
 [12.38505809, 22.55786533],
 [10.05918886, 25.09517722],
 [ 9.21341823, 25.94094784],
 [ 7.94476228, 28.05537442],
 [ 5.61889305, 31.64989959]]

array_path = []
array_path.append([[394.93484225 ,895.70472874],
 [404.37242798, 896.41254767],
 [409.09122085 ,897.1203666 ],
 [410.50685871, 897.82818553],
 [419.00068587, 899.71570268],
 [427.49451303 ,904.43449555],
 [431.74142661 ,907.02983162],
 [433.39300412 ,909.15328841],
 [436.46021948 ,915.28771914],
 [438.34773663 ,919.53463272],
 [441.65089163, 926.61282202],
 [441.65089163 ,934.39883025],
 [439.99931413 ,937.46604562],
 [439.52743484, 937.70198526]])

array_path.append([[392.57544582 ,902.31103876],
 [395.17078189 ,902.5469784 ],
 [397.05829904 ,903.72667662],
 [399.88957476 ,904.1985559 ],
 [411.92249657 ,906.55795234],
 [414.04595336 ,907.02983162],
 [418.52880658 ,908.20952984],
 [420.88820302 ,910.56892627],
 [426.7866941  ,915.0517795 ],
 [428.91015089 ,919.53463272],
 [429.38203018 ,923.7815463 ],
 [430.32578875 ,927.55658059],
 [430.79766804 ,937.9379249 ]])

path_interp, tangent =smutil.two_stage_interpolate(cam, delta_eval=2., smoothing=0.1, k=3)

normals = smutil.get_normals(tangent)


teste = smutil.dist(array_path[0][0], array_path[0][1])


vor, idx_medial_vertices, point_relation = smutil.medial_voronoi_ridges(array_path[0], array_path[1])
print(idx_medial_vertices)
print(type(idx_medial_vertices))

ver = smutil.order_ridge_vertices(idx_medial_vertices)

print(ver)
print(type(ver))

#print(idx_medial_vertices)
#print(point_relation)
