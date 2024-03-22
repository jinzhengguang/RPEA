# remove low confidence detections from the submission file
import os
import numpy as np

THRESH = 1e-3
SUBMISSION_DIR = "detections"
OUT_DIR = "{}_new_{}".format(SUBMISSION_DIR, THRESH)
print(OUT_DIR)

paths = os.listdir(SUBMISSION_DIR)
mymax = []
for i, path in enumerate(paths):
    print(f"process [{i}/{len(paths)}] {path}")
    seq_path = os.path.join(SUBMISSION_DIR, path)
    seq_path_new = os.path.join(OUT_DIR, path)
    os.makedirs(seq_path_new, exist_ok=True)

    for file in os.listdir(seq_path):
        det_path = os.path.join(seq_path, file)
        with open(det_path, "r") as f:
            lines = f.readlines()
        # confs = np.asarray([float(line.strip("\n").split(" ")[-1]) for line in lines])
        # lines_new = np.asarray(lines)[confs > THRESH]
        lines_new = []
        for line in lines:
            datai = line.strip("\n").split(" ")
            if float(datai[-1]) > THRESH:
                # Pedestrian 0 0 -1 0 -1 -1 -1 -1 1.7871 0.5645 0.9893 1.5605 0.7295 -3.0039 -2.7734 0.8789
                linei = 'Pedestrian 0 0 -1 0 -1 -1 -1 -1 {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    float(datai[9]), float(datai[10]), float(datai[11]), float(datai[12]),
                    float(datai[13]), float(datai[14]), float(datai[15]), float(datai[16]))
                lines_new.append(linei)
        lines_new[-1] = lines_new[-1].strip('\n')
        # print(len(lines), len(lines_new))
        mymax.append(len(lines_new))

        det_path_new = os.path.join(seq_path_new, file)
        with open(det_path_new, "w") as f:
            f.writelines(np.asarray(lines_new).tolist())

print('Pedestrian max: {}'.format(max(mymax)))
# 2023-04-03 Jinzheng Guang
