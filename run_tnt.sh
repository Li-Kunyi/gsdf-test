python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Barn -m outputs/tnt/Barn -r 2 --eval --save_ckpt #--ckpt_pth outputs/tnt/Barn/ckpt/ckpt_15000.pth
python extract_mesh_gsdf.py -m outputs/tnt/Barn --iteration 30000
python extract_mesh_mc.py -m outputs/tnt/Barn --iteration 30000
python eval_tnt/run.py --dataset-dir eval_tnt/TrainingSet/Barn --traj-path eval_tnt/TrainingSet/Barn/Barn_COLMAP_SfM.log --ply-path outputs/tnt/Barn/test/ours_30000/fusion/mesh_binary_search_7.ply

# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Caterpillar -m outputs/tnt/Caterpillar -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Courthouse -m outputs/tnt/Courthouse -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Ignatius -m outputs/tnt/Ignatius -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Meetingroom -m outputs/tnt/Meetingroom -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Truck -m outputs/tnt/Truck -r 2 --eval --save_ckpt


# load from ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Barn -m outputs/tnt/Barn -r 2 --eval --save_ckpt #--den_interval 500
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Caterpillar -m outputs/tnt/Caterpillar -r 2 --eval 
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Courthouse -m outputs/tnt/Courthouse -r 2 --eval
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Ignatius -m outputs/tnt/Ignatius -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Meetingroom -m outputs/tnt/Meetingroom -r 2 --eval --save_ckpt
# python train_sdf_v4.py -s /home/kunyi/work/data/tnt/TrainingSet/Truck -m outputs/tnt/Truck -r 2 --eval --save_ckpt



