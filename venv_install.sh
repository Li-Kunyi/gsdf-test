conda create -y -n gsdf python=3.8
conda activate gsdf

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# tetra-nerf for triangulation, optional
cd submodules/tetra-triangulation
conda install -y cmake
conda install -y conda-forge::gmp
conda install -y conda-forge::cgal
cmake .
make 
pip install -e .