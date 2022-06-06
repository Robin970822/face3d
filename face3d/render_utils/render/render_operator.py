#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import time
import yaml
import imageio
import argparse
import numpy as np
import os.path as osp

from glob import glob
from easydict import EasyDict

sys.path.append('../')
from face3d.render_utils.render.lighting import RenderPipeline

cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class Render(object):
    """docstring for Render"""

    def __init__(self):
        super(Render, self).__init__()
        self.app = RenderPipeline(None, **cfg)

    def __call__(self, image, vertices, triangles):
        vertices = vertices.astype(np.float32)
        vertices = vertices.copy(order='C')
        vertices[:, 2] = - vertices[:, 2]
        vertices[:, 2] -= np.min(vertices[:, 2])
        image = image.astype(np.float32) / 255.
        img_render = self.app(vertices, triangles, image)

        return img_render


def render_image(image, vertices, nb_kpts):
    app = RenderPipeline(None, **cfg)
    triangles = np.loadtxt(f'utils/triangle_data/face_{nb_kpts}pts.txt').astype(np.int32) - 1
    vertices = vertices.astype(np.float32)
    vertices = vertices.copy(order='C')
    vertices[:, 2] = - vertices[:, 2]
    vertices[:, 2] -= np.min(vertices[:, 2])
    image = image.astype(np.float32) / 255.
    img_render = app(vertices, triangles, image)

    return img_render


def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(
        description='3D mesh rendering')
    new_parser.add_argument('--exp_name', default='exp12')
    new_parser.add_argument('--nb_kpts', default='1945')

    return new_parser.parse_args()


def main():
    args = parse_args()
    exp_name = args.exp_name
    nb_kpts = args.nb_kpts

    data_root = f'../snapshot/{exp_name}/'

    out_dir = data_root + 'vis_dense/'
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    app = RenderPipeline(**cfg)

    mesh_info_path = data_root + 'mesh_results/'
    image_path = data_root + 'vis_results'
    img_fps = sorted(glob(f'{image_path}/*.jpg'))

    triangles = np.loadtxt(f'./our_data/face_{nb_kpts}pts.txt').astype(np.int32) - 1

    for img_fp in img_fps[:]:
        im_name = img_fp.split('/')[-1]
        mesh_path = mesh_info_path + im_name.replace('.jpg', '.txt')
        vertices = np.loadtxt(mesh_path).astype(np.float32)  # mx3
        vertices[:, 2] = - vertices[:, 2]
        vertices[:, 2] -= np.min(vertices[:, 2])

        img = imageio.imread(img_fp).astype(np.float32) / 255.

        img_render = app(vertices, triangles, img)

        img_wfp = osp.join(out_dir, osp.basename(img_fp))
        imageio.imwrite(img_wfp, img_render)
        print('Writing to {}'.format(img_wfp))


if __name__ == '__main__':
    main()
