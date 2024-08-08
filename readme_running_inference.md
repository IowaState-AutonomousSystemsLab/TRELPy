To run BevFusion:

nohup python3 -u tools/test.py projects/BEVFusion/configs/bevfusion
_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py  checkpoints/bevfusion_converted.pth  --cfg-options "val_evaluator
.pklfile_prefix=/home/apurvabadithela/nuscenes_dataset/inference_results/bevfusion_model/results.pkl" --task 'multi-modality_de
t' </dev/null > bev_test.out &
