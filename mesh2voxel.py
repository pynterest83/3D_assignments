import kaolin as kal
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def load_obj(file_path):
    mesh = kal.io.obj.import_mesh(file_path)
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    return vertices, faces

def voxelize_mesh(vertices, faces, voxel_size=32):
    vertices_tensor = torch.from_numpy(vertices).float().cuda()
    faces_tensor = torch.from_numpy(faces).long().cuda()
    voxel_grid = kal.ops.conversions.trianglemeshes_to_voxelgrids(
        vertices_tensor.unsqueeze(0), faces_tensor, voxel_size
    )
    return voxel_grid[0].cpu().numpy()

def plot_voxel(voxel_grid, angle):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, edgecolor='k')
    ax.view_init(elev=20, azim=angle)
    ax.set_axis_off()
    plt.tight_layout()
    
    temp_file = f"temp_{angle}.png"
    plt.savefig(temp_file, bbox_inches='tight', dpi=100)
    plt.close()
    return temp_file

def create_gif(voxel_grid, output_file="voxel_animation.gif", num_frames=36):
    images = []
    angles = np.linspace(0, 360, num_frames, endpoint=False)
    
    for angle in angles:
        temp_file = plot_voxel(voxel_grid, angle)
        images.append(imageio.imread(temp_file))
        os.remove(temp_file)
    
    imageio.mimsave(output_file, images, duration=0.1)
    print(f"GIF saved as {output_file}")

def main(obj_file_path):
    vertices, faces = load_obj(obj_file_path)
    voxel_grid = voxelize_mesh(vertices, faces)
    create_gif(voxel_grid)

if __name__ == "__main__":
    obj_file_path = "data/cow.obj"  
    main(obj_file_path)