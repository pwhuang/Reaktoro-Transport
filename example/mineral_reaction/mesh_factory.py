# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh
import numpy as np

class MeshFactory(MarkedRectangleMesh):
    def __init__(self, nx, ny, mesh_type='triangle', mesh_shape='left/right'):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 100.0, coord_y = 100.0)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type(mesh_type)
        self.locate_and_mark_boundaries()

        self.generate_mesh(mesh_shape)

        self.generate_boundary_markers()
        self.generate_interior_markers()
        self.generate_domain_markers()

    def locate_and_mark_boundaries(self, boundary_eps=1e-5):
        right_marker = lambda x: np.isclose(x[0], self.xmax, atol=boundary_eps)
        top_marker = lambda x: np.isclose(x[1], self.ymax, atol=boundary_eps)
        left_marker = lambda x: np.isclose(x[0], self.xmin, atol=boundary_eps)
        bottom_marker = lambda x: np.isclose(x[1], self.ymin, atol=boundary_eps)

        self.marker_dict = {"right": 1, "top": 2, "left": 3, "bottom": 4}

        self.locator_dict = {
            "right": right_marker,
            "top": top_marker,
            "left": left_marker,
            "bottom": bottom_marker
        }

        return self.marker_dict, self.locator_dict
