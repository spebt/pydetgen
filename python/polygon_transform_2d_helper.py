import torch

def rotate_polygons(polygons, angle):
    verts_tensor = polygons.view(-1, 2)
    rot_mat = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    return (
        torch.bmm(
            rot_mat.unsqueeze(0).expand(verts_tensor.shape[0], -1, -1),
            verts_tensor.unsqueeze(-1),
        )
        .squeeze(-1)
        .view(polygons.shape)
    )