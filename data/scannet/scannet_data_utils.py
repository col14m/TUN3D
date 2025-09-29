# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp
import json
import mmengine
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(os.path.dirname(FILE_DIR), 'data/scannet')
LAYOUT_ANNOTATION_DIR = os.path.join(BASE_DIR, 'scannet_planes/')
DATA_DIR = os.path.join(BASE_DIR, 'points')


def get_boundary_coords(verts):
    x_diff = verts[:, 0].max() - verts[:, 0].min()
    y_diff = verts[:, 1].max() - verts[:, 1].min()
    coord = verts[:, 0] if x_diff > y_diff else verts[:, 1]
    min_ind = np.argmin(coord)
    max_ind = np.argmax(coord)
    if min_ind > max_ind:
        min_ind, max_ind = max_ind, min_ind
    return [list(verts[min_ind]), list(verts[max_ind])]


def approx_quad(verts):
    if len(verts) == 4:
        return verts

    verts = np.array(verts)
    h = verts[:, 2]
    thr = 0.5 * (h.min() + h.max())
    floor_verts = verts[h < thr]
    ceiling_verts = verts[h >= thr]
    boundary_verts_floor = get_boundary_coords(floor_verts)
    boundary_verts_ceiling = get_boundary_coords(ceiling_verts)
    return boundary_verts_floor + boundary_verts_ceiling


def isFourPointsInSamePlane(p0, p1, p2, p3, error):
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    s1 = p1-p0
    s2 = p2-p0
    s3 = p3-p0
    result = s1[0]*s2[1]*s3[2]+s1[1]*s2[2]*s3[0]+s1[2]*s2[0] * \
        s3[1]-s1[2]*s2[1]*s3[0]-s1[0]*s2[2]*s3[1]-s1[1]*s2[0]*s3[2]
    if result - error <= 0 <= result + error:
        return True
    return False


def get_normal(quad_vert, center):
    tmp_A = []
    tmp_b = []
    for i in range(4):
        tmp_A.append([quad_vert[i][0], quad_vert[i][1], 1])  # x,y,1
        tmp_b.append(quad_vert[i][2])  # z
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    temp = A.T * A
    if np.linalg.det(temp) > 1e-10:
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0]/fit[2][0]
        b = fit[1][0]/fit[2][0]
        c = -1.0/fit[2][0]
        normal_vector = np.array([a, b, c])

        # print ("solution:%f x + %f y + %f z + 1 = 0" % (a, b, c) )

    else:  # vertical
        b = np.matrix([-1, -1, -1, -1]).T
        A = A[:, 0:2]
        temp = A.T * A
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0]
        b = fit[1][0]
        c = 0
        normal_vector = np.array([a, b, c])
        # print ("solution:%f x + %f y + 1 = 0" % (a, b) )

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector


def transform(scan_name, mesh_vertices):
    meta_file = BASE_DIR + '/scans_transform/scans/' + \
        os.path.join(scan_name, scan_name+'.txt')
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x)
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    return mesh_vertices


def get_center(verts):
    verts = np.array(verts)
    center = np.mean(verts, axis=0)
    return center


def get_quads(scan_name, approx_non_quads=False):
    with open(LAYOUT_ANNOTATION_DIR + scan_name+'.json', 'r') as quad_file:
        plane_dict = json.load(quad_file)
    quad_dict = plane_dict['quads']
    overall_n_quads = len(quad_dict)
    vert_dict = plane_dict['verts']

    for i in range(0, len(vert_dict)):
        temp = vert_dict[i][1]
        vert_dict[i][1] = - vert_dict[i][2]
        vert_dict[i][2] = temp

    verts = np.array(vert_dict)

    verts = transform(scan_name, verts)

    def get_last(arr):
        return arr[-1]

    if not approx_non_quads:
        quads = [i for i in quad_dict if len(i) == 4]
        quad_verts = [sorted([list(verts[j])
                             for j in _], key=get_last) for _ in quads]

    else:
        quad_verts = [sorted(approx_quad([list(verts[j])
                             for j in q]), key=get_last) for q in quad_dict]

    quad_verts_filter_ = [list(quad_vert) for quad_vert in quad_verts
                          if isFourPointsInSamePlane(quad_vert[0], quad_vert[1], quad_vert[2], quad_vert[3], 100)]

    room_center = get_center(vert_dict)  # room center

    vert_quads = [list(quad_vert) for quad_vert in quad_verts_filter_
                  if abs(get_normal(quad_vert, room_center)[2]) < 0.2]  # only vertical

    horizontal_quads = [list(quad_vert) for quad_vert in quad_verts_filter_
                        if abs(get_normal(quad_vert, room_center)[2]) > 0.8]  # only horizontal

    return vert_quads, horizontal_quads, overall_n_quads
