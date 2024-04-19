from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class ConformalLoss:
    def __init__(self, parameters, shape_groups):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.parameters = parameters
        self.target_letter = ["C"]
        self.shape_groups = shape_groups
        self.faces = self.init_faces(self.device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()

    def get_angles(self, points):
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters])
        self.angles = self.get_angles(points)

    def init_faces(self, device):
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters[i].clone().detach().cpu().numpy() for i in
                         range(len(self.parameters))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind + 1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self):
        loss_angles = 0
        points = torch.cat(self.parameters)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles
