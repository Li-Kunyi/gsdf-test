# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Barn -m outputs/tnt_test/Barn -r 2 --eval --save_ckpt --ckpt_pth outputs/tnt_test/Barn/ckpt/ckpt_25000.pth
python extract_mesh_gsdf.py -m outputs/tnt_test/Barn --iteration 30000
python extract_mesh_mc.py -m outputs/tnt_test/Barn --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path eval_tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt_test/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_v4.py -s /mnt/user/datasets/tnt/TrainingSet/Caterpillar -m outputs/tnt_node07/Caterpillar -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/tnt/TrainingSet/Courthouse -m outputs/tnt_node07/Courthouse -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/tnt/TrainingSet/Ignatius -m outputs/tnt_node07/Ignatius -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/tnt/TrainingSet/Meetingroom -m outputs/tnt_node07/Meetingroom -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/tnt/TrainingSet/Truck -m outputs/tnt_node07/Truck -r 2 --eval --save_ckpt

# python train_sdf_v4.py -s /mnt/user/datasets/360/garden -m outputs/360_node07/garden -r 4 --eval --save_ckpt --den_interval 500
# python train_sdf_v4.py -s /mnt/user/datasets/360/bicycle -m outputs/360_node07/bicycle -r 4 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/360/bonsai -m outputs/360_node07/bonsai -r 2 --eval --save_ckpt --den_interval 500
# python train_sdf_v4.py -s /mnt/user/datasets/360/counter -m outputs/360/counter -r 2 --eval --save_ckpt
# # python train_sdf_v4.py -s /mnt/user/datasets/360/flowers -m outputs/360/flowers -r 4 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/360/stump -m outputs/360/stump -r 4 --eval --save_ckpt
# # python train_sdf_v4.py -s /mnt/user/datasets/360/treehill -m outputs/360/treehill -r 4 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/360/kitchen -m outputs/360/kitchen -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /mnt/user/datasets/360/room -m outputs/360/room -r 2 --eval --save_ckpt
