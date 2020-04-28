import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
)
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.cameras import SfMPerspectiveCameras, SfMOrthographicCameras

device = torch.device("cuda:0")
torch.cuda.set_device(device)

obj_filename = 'ra_data/out.obj'

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)
texture_image=mesh.textures.maps_padded()
plt.figure(figsize=(7, 7))
plt.imshow(texture_image.squeeze().cpu().numpy())

cam_t = torch.tensor([-0.0578,  0.1563, 39.0994]).view(1, -1).to(device)
cameras = SfMPerspectiveCameras(focal_length=torch.ones(1, 1)*5000, principal_point=torch.tensor([112, 112])[None], device=device, T=cam_t)

raster_settings = RasterizationSettings(
    image_size=224,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=TexturedSoftPhongShader(
        device=device,
        cameras=cameras,
    )
)

images = renderer(mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.grid("off")
plt.axis("off")
