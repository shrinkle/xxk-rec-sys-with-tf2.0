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

vid_emb_matrix_norm = np.load("vid_emb_matrix_norm.npy", allow_pickle=True)
vid = np.load("vid.npy", allow_pickle=True)

#faiss 检索
res = faiss.StandardGpuResources()
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0

try:
    gpu_index = faiss.GpuIndexFlatIP(res, k, flat_config)
    gpu_index.add(vid_emb_matrix_norm)
except Exception as e:
    print(e)

daytime = time.strftime('%m%d', time.localtime(time.time()))

_, index_matrix = gpu_index.search(vid_emb_matrix_norm, topN)
vid_haddle = open(HOME_PATH + "/result/dcnv2_vid_rec_" + daytime, 'w')
for i in range(len(vid_emb_matrix_norm)):
    rec_vid = vid[index_matrix[i]]
    vid_haddle.write(vid[i] + "#2\t" + "#2,".join(rec_vid) + "#2\n")

