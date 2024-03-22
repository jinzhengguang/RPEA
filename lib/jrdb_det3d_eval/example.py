from jrdb_det3d_eval import eval_jrdb

det_dir = "detections"
gt_dir = "labels_kitti"

ap_dict = eval_jrdb(gt_dir, det_dir)
for k, v in ap_dict.items():
    print(k, v)
