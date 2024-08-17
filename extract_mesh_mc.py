import torch
import os
from os import makedirs
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
from scene.sdf_gaussian_model_v3 import GaussianModel
import numpy as np
import trimesh
from skimage.measure import marching_cubes
import marching_cubes as mcubes


@torch.no_grad()
def marching_cube(model_path, name, iteration, gaussians):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")

    makedirs(render_path, exist_ok=True)
    
    max_bound = gaussians.bounding_box[:, 1] / 12
    min_bound = gaussians.bounding_box[:, 0] / 12
    vox_size = 0.01
    grid_size = ((max_bound - min_bound) / vox_size).long() + 1  # [D, H, W]

    # Function to generate grid points in batches
    def generate_grid_points(min_bound, max_bound, grid_size, batch_size):
        x = torch.linspace(min_bound[0], max_bound[0], grid_size[0])
        y = torch.linspace(min_bound[1], max_bound[1], grid_size[1])
        z = torch.linspace(min_bound[2], max_bound[2], grid_size[2])
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)
        flat_z = grid_z.reshape(-1)

        num_vox = flat_x.shape[0]
        
        for i in range(0, num_vox, batch_size):
            start = i
            end = min(i + batch_size, num_vox)
            yield torch.stack((flat_x[start:end], flat_y[start:end], flat_z[start:end]), dim=-1).cuda()

    # Query SDF and process in batches
    batch_size = 100_000
    sdfs = []
    for grid_batch in tqdm(generate_grid_points(min_bound, max_bound, grid_size, batch_size), desc="Marching Cube"):
        _sdf = gaussians.query_sdf(grid_batch)['sdf']
        _sdf = _sdf.detach().cpu().numpy()
        sdfs.append(_sdf)

    sdfs = np.concatenate(sdfs, axis=0)
    sdf_grid = sdfs.reshape(grid_size.tolist())
    print('Running Marching Cubes')
    # verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0)
    verts, faces = mcubes.marching_cubes(sdf_grid, 0.0, truncation=3.0)
    print('done', verts.shape, faces.shape)

    mesh = trimesh.Trimesh(verts, faces, process=False)
    # get connected components
    # components = mesh.split(only_watertight=False)
    # if False:
    #     areas = np.array([c.area for c in components], dtype=np.float16)
    #     mesh = components[areas.argmax()]
    # else:
    #     new_components = []
    #     for comp in components:
    #         if comp.area > 1.0:
    #             new_components.append(comp)
    #     mesh = trimesh.util.concatenate(new_components)
    # verts = mesh.vertices
    # faces = mesh.faces
    # mesh = trimesh.Trimesh(verts, faces, process=False)

    mesh.export(os.path.join(render_path, f"mesh_marching_cube_{iteration}.ply"))
    print('Mesh saved')
    

def extract_mesh(dataset : ModelParams, opt, iteration : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, opt.network)
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        gaussians.load_model(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "model.pt"))
        
        marching_cube(dataset.model_path, "test", iteration, gaussians)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=10000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hash_size", type=int, default=22)
    parser.add_argument("--hash_resolution", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = get_combined_args(parser)

    op.network['grid']['hash_size'] = args.hash_size
    op.network['grid']['resolution'] = args.hash_resolution
    op.network['grid']['alpha'] = args.alpha

    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), op.extract(args), args.iteration)