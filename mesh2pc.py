import argparse
import pytorch3d
import torch

import imageio

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def load_cow_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces

def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def mesh2pc(
    cow_path = 'data/cow.obj',
    image_size=256,
    device=None,
    num_samples=10000,
    background_color=(1, 1, 1),
):
    if device is None:
        device = get_device()
    
    vertices, faces = load_cow_mesh(cow_path)
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    # breakpoint()
    vertices = torch.tensor(vertices)
    cross_product = torch.cross(v1-v0,v2-v0,dim=1)
    areas = torch.norm(cross_product,dim=1)/2
    probs = areas / areas.sum()
    breakpoint()
    face_indices = torch.multinomial(probs, num_samples, replacement=True)

    # Sample points inside selected triangles using barycentric coordinates
    u = torch.sqrt(torch.rand(num_samples))
    v = torch.rand(num_samples)
    w = 1 - u
    u = u * (1 - v)
    v = u * v

    # Interpolate points
    sampled_points = (
        w.unsqueeze(1) * v0[face_indices] +
        u.unsqueeze(1) * v1[face_indices] +
        v.unsqueeze(1) * v2[face_indices]
    ).to(device).unsqueeze(0)
    # breakpoint()
    # vertices = vertices.unsqueeze(0)
    # breakpoint()
    # point_cloud = torch.cat([vertices.to(device),sampled_points],dim=0)
    renderer = get_points_renderer(image_size=image_size,background_color=background_color)
    textures = torch.ones_like(sampled_points)
    textures = textures * torch.tensor([0.6,0.7,1], device=device)
    # textures = textures
    point_cloud = pytorch3d.structures.Pointclouds(
        points = sampled_points,
        features = textures
    )
    images=[]
    for azimuth in torch.linspace(0,360, 200):
        R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist=2,\
            elev = 0, azim = azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,T=T, fov=60, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0,0,-3]], device=device)
        
        rend = renderer(point_cloud, cameras = cameras, lights = lights)
        image= rend.cpu().numpy()[0,...,:3]# (B, H, W, 4) -> (H, W, 3)
        images.append((image*255).astype('uint8'))
    return images

def save_gif(images, output_path='cow.gif', fps=15):
    imageio.mimsave(output_path,images, fps = fps)
    print("save successfully gif")
    
    
if __name__ =="__main__":
    images = mesh2pc()
    save_gif(images,'output/mesh2pc.gif')