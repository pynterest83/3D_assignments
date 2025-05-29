import os
import torch
import imageio # For creating the GIF
from tqdm import tqdm # For a progress bar

# --- PyTorch3D imports ---
from pytorch3d.io import load_obj # Or load_ply, depending on your mesh file
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
# If the cow mesh is part of PyTorch3D's datasets (use if no file is provided in the repo)
# from pytorch3d.datasets import fetch_cow_mesh

# --- 1. Set up device (CPU in your case) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- 2. Load the Cow Mesh ---
# IMPORTANT: Replace with the actual path to your cow mesh file
# mesh_path = "path/to/your/cow.obj" # e.g., "data/cow.obj" or "task1/cow.obj"

# If the mesh is provided as a file in the repository:
# Check common locations. This is a placeholder, adjust as needed.
cow_mesh_path_in_repo = "data/cow.obj" # Example path, CHECK YOUR REPO
if os.path.exists(cow_mesh_path_in_repo):
    verts, faces_idx, _ = load_obj(cow_mesh_path_in_repo)
    faces = faces_idx.verts_idx
else:
    # Fallback: If you can't find a specific cow mesh file in the repository,
    # and if PyTorch3D's example cow is acceptable for the assignment:
    print(f"Cow mesh not found at {cow_mesh_path_in_repo}. Using PyTorch3D example cow.")
    # Make sure you have an internet connection the first time you run this
    # It will download the cow mesh to a local cache.
    # verts, faces = fetch_cow_mesh(device=device)
    # If fetch_cow_mesh is not directly available or you prefer to load manually after download:
    # You might need to download 'cow.obj' from PyTorch3D's GitHub or examples
    # and place it in your project, then use load_obj.
    # For simplicity, let's assume you need to find the file from the assignment repo.
    # If you absolutely cannot find it, ask your instructor or use a placeholder.
    print("ERROR: Cow mesh file not found. Please locate the 'provided cow mesh' from the assignment.")
    exit()


# Initialize faces with a texture (e.g., white color)
# Each vertex will have a color. Let's make it white.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

# Create a PyTorch3D Meshes object.
# A mesh is composed of vertices, faces, and optionally textures.
mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)

# --- 3. Initialize a Renderer ---
# The snippet mentions `renderer(...)` but doesn't show its setup.
# We need a rasterizer and a shader.
image_size = 512  # Output image size (height and width)

# Initialize a perspective camera.
# `look_at_view_transform` will be used to set R and T for each frame.
# The FOV (Field of View) and other camera params are set here.
# These R and T are initial values and will be overridden in the loop.
R_init, T_init = look_at_view_transform(dist=2.7, elev=0, azim=0)
cameras_init = FoVPerspectiveCameras(device=device, R=R_init, T=T_init, fov=60)

# Define the settings for rasterization and shading.
# For example, using a Phong shader to get basic shading.
raster_settings = RasterizationSettings(
    image_size=image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Define lights. The snippet used PointLights.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]]) # Adjusted z for cow visibility

# Create a Phong renderer by composing a rasterizer and a shader.
# The Phong shader uses lights and camera position.
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras_init, # Will be updated per frame
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device,
        cameras=cameras_init, # Will be updated per frame
        lights=lights
    )
)

# --- 4. Render a GIF from different viewpoints ---
images = []
num_images = 60  # Number of frames in the GIF
dist_val = 2.7    # Distance of the camera from the object
elev_val = 20.0   # Elevation of the camera (degrees)

print(f"Rendering {num_images} frames for the GIF...")
for i in tqdm(range(num_images)):
    azimuth = torch.linspace(0, 360, num_images + 1)[i] # Ensure full 360, avoid duplicate end frame if using num_images directly in linspace
    
    # Get rotations and translations for the current camera viewpoint
    R, T = look_at_view_transform(dist=dist_val, elev=elev_val, azim=azimuth)
    
    # Create a new camera for this viewpoint
    # Note: fov, aspect_ratio, etc., are taken from cameras_init if not specified
    current_cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)
    
    # Render the mesh.
    # The renderer uses the mesh, current camera, and lights.
    # Note: The snippet used `renderer(point_cloud, ...)` but task is "cow mesh".
    # Assuming MeshRenderer, it typically takes `meshes_world` and keyword args.
    rend = renderer(mesh, cameras=current_cameras, lights=lights)
    
    # Convert the rendered image to a NumPy array for imageio
    # rend is (B, H, W, 4) - (Batch, Height, Width, RGBA)
    # We take the first image in the batch and only RGB channels.
    image = rend[0, ..., :3].cpu().numpy() # (H, W, 3)
    
    # Convert to uint8 format (0-255)
    images.append((image * 255).astype('uint8'))

# --- 5. Save the GIF ---
gif_filename = 'cow_turntable.gif'
imageio.mimsave(gif_filename, images, fps=15)
print(f"GIF saved as {gif_filename}")