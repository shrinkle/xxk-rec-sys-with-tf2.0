import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing
import time
import torch
from collections import Counter
import faiss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

HOME_PATH = '/data/app/xxk/RST2/DCNV2'
read_part = True
sample_num = 20000000
test_size = 0.2
k = 128
topN = 200
model_name = "DCN-Mix"
model = torch.load(HOME_PATH + '/' + model_name + '.h5', map_location='cpu')
param = model.state_dict()
np.save("param", param)
feature_columns = np.load(HOME_PATH + "/feature_columns.npy", allow_pickle=True)
keys = ['embedding_dict.' + line.name + '.weight' for line in feature_columns if 'sparse_feature_v' in line.name]
np.save("keys", keys)
daytime = time.strftime('%m%d', time.localtime(time.time()))

df = np.load(HOME_PATH + "/data_df.npy", allow_pickle=True)
counter = Counter()
counter.update(df[:, 2])
valid_vid = set()
for k, v in counter.items():
    if v > 10:
        valid_vid.add(k)


predict_uid, predict_vid, predict_uid2vid = np.load(HOME_PATH + "/predict.npy", allow_pickle=True)

vid, vid_dense_feat, vid_sparse_feat = predict_vid
vid = np.reshape(vid, -1)
vid_emb_matrix = []
uid_feat_length = predict_uid[1].shape[1]
dense_feat_length = 2

temp_vid = []
print("start build vid emb matrix!")
for i in range(vid.shape[0]):
    if int(vid[i]) not in valid_vid:
        continue
    # temp_vid.append(vid[i])
    vid_all_feat_emb = []
    for m, n in enumerate(vid_sparse_feat[i]):
        if n == 0:
            continue
        temp_emb = param['embedding_dict.' + feature_columns[m + uid_feat_length + dense_feat_length].name + '.weight'][n].cpu().numpy().astype('float32')
        vid_all_feat_emb.append(temp_emb)
    if len(vid_all_feat_emb) > 0:
        vid_emb_matrix.append(np.mean(vid_all_feat_emb, axis=0))
        temp_vid.append(vid[i])
    # else:
    #     vid_emb_matrix.append(np.zeros((128, ), dtype=np.float32))

vid_emb_matrix_norm = preprocessing.normalize(vid_emb_matrix, norm='l2').astype('float32')
vid = temp_vid
np.save(HOME_PATH + "/vid_emb_matrix_norm", vid_emb_matrix_norm)
np.save(HOME_PATH + "/vid", vid)

# 保存vid_vector
print("start build vid vector!")
daytime = time.strftime('%m%d', time.localtime(time.time()))
with open(HOME_PATH + "/result/video_vector." + daytime, "w") as f:
    for i in range(len(vid)):
        f.write(str(vid[i]) + "2\t" + ",".join(map(str, vid_emb_matrix_norm[i])) + "\n")


