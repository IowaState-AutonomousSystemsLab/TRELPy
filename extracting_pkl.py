import pickle
import pdb 

def read_pkl_file(fn='nuscenes_infos_val.pkl'):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    nusc_fn = "/home/apurvabadithela/software/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl"
    demo_pkl = "/home/apurvabadithela/software/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl"
    demo_data = read_pkl_file(demo_pkl)
    full_val_nusc = read_pkl_file(nusc_fn)
    pdb.set_trace()
    
    # test_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_val.pkl', pklfile_prefix=None, submission_prefix=None)
    # --cfg-options test_evaluator.pklfile_prefix=${RESULT_FILE}

    # python tools/test.py projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    # checkpoints/bevfusion_converted.pth --cfg-options 'test_evaluator.pklfile_prefix=/home/apurvabadithela/nuscenes_dataset/test/bev.pkl' \ 
    # --show-dir ./data/nuscenes/show_results