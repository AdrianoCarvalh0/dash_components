import math
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import Voronoi
from shapely import geometry, ops as shops

def arc_length(path):
    """Calculate the cumulative arc-length between points in path"""
    
    dl = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    l = np.cumsum(dl).tolist()
    l = np.array([0] + l)
    
    return l

def interpolate(path, delta_eval=2., smoothing=0.1, k=3, return_params=False):
    """Interpolate a list of points"""

    path = np.array(path)
    l = arc_length(path)

    num_points = len(path)
    (tck, u), fp, ier, msg = splprep(path.T, s=smoothing*num_points, k=k, full_output=True)
    
    delta_eval_norm = delta_eval/l[-1]
    eval_points = np.arange(0, 1+0.75*delta_eval_norm, delta_eval_norm)
    x_interp, y_interp = splev(eval_points, tck, ext=3)
    dx_interp, dy_interp = splev(eval_points, tck, der=1, ext=3)
    
    path_interp = np.array([x_interp, y_interp]).T
    tangent = np.array([dx_interp, dy_interp]).T
    t_norm = np.sqrt(np.sum(tangent**2, axis=1))
    tangent = tangent/t_norm[None].T
    
    if return_params:
        return path_interp, tangent, tck, u
    else:
        return path_interp, tangent

def two_stage_interpolate(path, delta_eval=2., smoothing=0.1, k=3):
    """Interpolate path in two stages. First, a linear interpolation is applied so that
    intermediate points are generated. Then, a cubic interpolation is applied. This is useful
    because the cubic interpolation guarantees that the spline passes near the original points
    in the path, but it can be far away from the original curve between two original points. By 
    first doing a linear interpolation followed by a cubic one, the resulting spline cannot be
    too far away from the original path.
    
    Parameters:
    -----------
    path : ndarray
        List of points containing the path to be inteprolated.
    delta_eval : float
        The interval to evaluate the interpolation.
    smoothing : float
        Smoothing factor. 0 means the the spline will pass through all linearly-interpolated points.
    k : int
        The degree of the second interpolation.
    """
    
    path_interp_linear, _ = interpolate(path, delta_eval=delta_eval, smoothing=0, k=1)
    path_interp, tangent = interpolate(path_interp_linear, delta_eval=delta_eval, smoothing=smoothing, k=k)
 
    return path_interp, tangent

def get_normals(tangents):
    """Get normal vectors from a list of tangent vectors"""
    
    normals = np.zeros((len(tangents), 2))
    for idx, t in enumerate(tangents):
        tx, ty = t
        if ty<1e-3:
            n2 = 1
            n1 = -ty*n2/tx
        else:
            n1 = 1
            n2 = -tx*n1/ty

        norm = np.sqrt(n1**2 + n2**2)
        n = np.array([n1/norm, n2/norm])
        
        orient = np.sign(np.cross(t, n))
        if idx>0:
            if orient!=prev_orient:
                # Orientation of vector is different from previous vector, flip it
                n *= -1
                orient *= -1
        prev_orient = orient
        normals[idx] = n
        
    return normals

def dist(p1, p2):
    """Distance between two points"""
    
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def medial_voronoi_ridges(path1, path2):
    """Extract medial Voronoi ridges between path1 and path2. Those can be used
    for representing the medial path of a tubular structure."""
    
    all_points = np.concatenate((path1, path2), axis=0)
    all_points_ordered = np.concatenate((path1, path2[::-1]), axis=0)
    vor = Voronoi(all_points)
    num_points_path1 = len(path1)
    tube_region = geometry.Polygon(all_points_ordered)
    idx_internal_vertices = set()
    # Get Voronoi vertices inside the tube
    for idx_vertex, vertex in enumerate(vor.vertices):
        if tube_region.contains(geometry.Point(vertex)):
            idx_internal_vertices.add(idx_vertex)
 
    idx_medial_vertices = []
    point_relation = []
    for idx, ridge in enumerate(vor.ridge_points):

        first_is_path1 = True if ridge[0] < num_points_path1 else False
        second_is_path1 = True if ridge[1] < num_points_path1 else False
        if (first_is_path1+second_is_path1)==1:
            # If ridge is between a point in path1 and another in path2
            idx_ridge_vertices = vor.ridge_vertices[idx]
            if idx_ridge_vertices[0] in idx_internal_vertices and idx_ridge_vertices[1] in idx_internal_vertices:
                # Be careful with -1 index in idx_ridge_vertices for terminal points
                idx_medial_vertices.append(idx_ridge_vertices)
                if ridge[0] < num_points_path1:
                    point_relation.append((ridge[0], ridge[1]))
                else:
                    point_relation.append((ridge[1], ridge[0]))

    idx_medial_vertices = np.array(idx_medial_vertices)
    point_relation = np.array(point_relation)
    
    return vor, idx_medial_vertices, point_relation

def order_ridge_vertices(idx_vertices):
    """Given a list of Voronoi ridge vertices that define a piecewise path but are unordered,
    order the ridges."""
    
    idx_vertices = list(map(tuple, idx_vertices))
    vertice_ridge_map = {}
    last_vertex = -1
    for idx_ridge, (idx_v1, idx_v2) in enumerate(idx_vertices):
        if idx_v1 in vertice_ridge_map:
            vertice_ridge_map[idx_v1].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v1] = [idx_ridge]

        if idx_v2 in vertice_ridge_map:
            vertice_ridge_map[idx_v2].append(idx_ridge)
        else:
            vertice_ridge_map[idx_v2] = [idx_ridge]
    
    for idx_vertex, indices_ridge in vertice_ridge_map.items():
        if len(indices_ridge)==1:
            idx_first_vertex = idx_vertex
            break
    
    ordered_vertices = [idx_first_vertex]
    idx_ridge = vertice_ridge_map[idx_first_vertex][0]
    idx_v1, idx_v2 = idx_vertices[idx_ridge]
    if idx_v1==idx_first_vertex:
        idx_vertex = idx_v2
    else:
        idx_vertex = idx_v1
    ordered_vertices.append(idx_vertex)
    prev_idx_ridge = idx_ridge
    prev_idx_vertex = idx_vertex
    while True:
        indices_ridge = vertice_ridge_map[idx_vertex]
        if len(indices_ridge)==1:
            break
        if indices_ridge[0]==prev_idx_ridge:
            idx_ridge = indices_ridge[1]
        else:
            idx_ridge = indices_ridge[0]
        idx_v1, idx_v2 = idx_vertices[idx_ridge]
        if idx_v1==prev_idx_vertex:
            idx_vertex = idx_v2
        else:
            idx_vertex = idx_v1
        
        ordered_vertices.append(idx_vertex)
        prev_idx_ridge = idx_ridge
        prev_idx_vertex = idx_vertex
    
    return ordered_vertices
        
def invert_if_oposite(path1, path2):
    """Invert path2 if path1 and path2 run on opposite directions"""
    
    min_size = min([len(path1), len(path2)])
    avg_dist = np.sum(np.sqrt(np.sum((path1[:min_size]-path2[:min_size])**2, axis=1)))
    avg_dist_inv = np.sum(np.sqrt(np.sum((path1[:min_size]-path2[::-1][:min_size])**2, axis=1)))
    if avg_dist_inv<avg_dist:
        # Paths go on opposite directions
        path2 = path2[::-1]
    
    return path2

def increase_path_resolution(path, res_factor):
    
    x, y = path.T
    num_points = len(path)
    indices = list(range(num_points))
    # Define parametric variable making sure that it passes through all the original points
    tck, _ = splprep(path.T, u=indices, s=0, k=3)
    eval_points = np.linspace(0, num_points-1, num_points*res_factor - (res_factor-1))
    x_interp, y_interp = splev(eval_points, tck, der=0)
    x_tangents, y_tangents = splev(eval_points, tck, der=1)
    
    path_interp = np.array([x_interp, y_interp]).T
    tangents = np.array([x_tangents, y_tangents]).T
    
    return path_interp, tangents

def find_point_idx(sh_path, point):
    
    dists = np.sqrt((sh_path.xy[0]-point[0])**2+(sh_path.xy[1]-point[1])**2)
    return np.argmin(dists)
    