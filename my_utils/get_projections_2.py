import numpy as np
import time
import open3d as o3d
import os
from PIL import Image
import cv2
import argparse
import sys


def background_crop(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray_img.shape)
    col = np.mean(gray_img, axis=0)
    row = np.mean(gray_img, axis=1)
    for i in range(len(col)):
        if col[i] != 255:
            col_a = i
            break
    for i in range(len(col)):
        if col[-i] != 255:
            col_b = len(col) - i
            break
    for i in range(len(row)):
        if row[i] != 255:
            row_a = i
            break
    for i in range(len(row)):
        if row[-i] != 255:
            row_b = len(row) - i
            break
    img = img[row_a:row_b, col_a:col_b, :]
    return img


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# Camera Rotation
def camera_rotation(type, path, img_path):
    print(path)
    if type == 'ply':
        point_cloud = o3d.io.read_point_cloud(path)
    elif type == 'mesh':
        obj = o3d.io.read_triangle_mesh(path, True)
        obj.compute_vertex_normals()

    if not os.path.exists(img_path + '/'):
        os.mkdir(img_path + '/')

    rotation_matrix = np.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0],
                                [0, 0, 0, 1]])

    for i in range(4):
        # 设置视角
        point_cloud = point_cloud.transform(rotation_matrix)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(point_cloud)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 1, 0])
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)

        image = background_crop(image)
        cv2.imwrite(os.path.join(img_path, str(i) + '.png'), image)
        vis.destroy_window()
        del ctr
        del vis


    # sys.exit(0)


def projection(type, path, img_path):
    # find all the objects 
    objs = os.walk(path)
    cnt = 0
    for path, dir_list, file_list in objs:
        cnt = cnt + 1
        for obj in file_list:
            # for textured mesh
            if type == 'mesh':
                # for tmq source
                if obj.endswith('.obj') in path:
                    one_object_path = os.path.join(path, obj)
                    camera_rotation(type, one_object_path, generate_dir(os.path.join(img_path, obj)))
                else:
                    continue
            # for colored point clouds
            elif type == 'ply':
                one_object_path = os.path.join(path, obj)
                camera_rotation(type, one_object_path, generate_dir(os.path.join(img_path, obj)))


if __name__ == '__main__':
    # capture the projections of the 3d model
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='ply')  # the format of the 3D model
    parser.add_argument('--path', type=str,
                        default='G:\database\LS_PCQA\distortionpc_all')  # path to the file that contain .ply models
    parser.add_argument('--img_path', type=str,
                        default='G:\database\LS_PCQA\LS_projections')  # path to the generated 2D input

    config = parser.parse_args()
    projection(config.type, config.path, config.img_path)
