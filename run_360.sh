python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/garden -m outputs/360/garden -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/bicycle -m outputs/360/bicycle -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/bonsai -m outputs/360/bonsai -r 2 --alpha 0.4 --hash_size 18 --hash_resolution 512 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/counter -m outputs/360/counter -r 2 --alpha 0.3 --hash_size 18 --hash_resolution 512 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/flowers -m outputs/360/flowers -r 4 --alpha 0.x --hash_size 18 --hash_resolution 512 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/stump -m outputs/360/stump -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/treehill -m outputs/360/treehill -r 4 --alpha 0.1 --hash_size 18 --hash_resolution 512 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/kitchen -m outputs/360/kitchen -r 2 --alpha 0.5 --hash_size 16 --hash_resolution 256 --eval --save_ckpt
python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/room -m outputs/360/room -r 2 --alpha 0.3 --hash_size 18 --hash_resolution 512 --eval --save_ckpt


# load from ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/garden -m outputs/360/garden -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt --ckpt_pth outputs/360/garden/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/bicycle -m outputs/360/bicycle -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt --ckpt_pth outputs/360/bicycle/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/bonsai -m outputs/360/bonsai -r 2 --alpha 0.4 --hash_size 18 --hash_resolution 512 --eval --save_ckpt --ckpt_pth outputs/360/bonsai/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/counter -m outputs/360/counter -r 2 --alpha 0.3 --hash_size 18 --hash_resolution 512 --eval --save_ckpt --ckpt_pth outputs/360/counter/ckpt/ckpt_<iteration>.pth 
# # python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/flowers -m outputs/360/flowers -r 4 --alpha 0.x --hash_size 18 --hash_resolution 512 --eval --save_ckpt --ckpt_pth outputs/360/flowers/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/stump -m outputs/360/stump -r 4 --alpha 0.05 --hash_size 20 --hash_resolution 1024 --eval --save_ckpt --ckpt_pth outputs/360/stump/ckpt/ckpt_<iteration>.pth 
# # python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/treehill -m outputs/360/treehill -r 4 --alpha 0.05 --hash_size 18 --hash_resolution 512 --eval --save_ckpt --ckpt_pth outputs/360/treehill/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/kitchen -m outputs/360/kitchen -r 2 --alpha 0.5 --hash_size 16 --hash_resolution 256 --eval --save_ckpt --ckpt_pth outputs/360/kitchen/ckpt/ckpt_<iteration>.pth 
# python train_sdf_v4.py -s /home/kunyi/work/data/360_v2/room -m outputs/360/room -r 2 --alpha 0.3 --hash_size 18 --hash_resolution 512 --eval --save_ckpt --ckpt_pth outputs/360/room/ckpt/ckpt_<iteration>.pth 

