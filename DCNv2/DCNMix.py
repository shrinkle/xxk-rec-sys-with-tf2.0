# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import DCNMix
from utils import check_model, get_train_data, SAMPLE_SIZE, get_device
import numpy as np


@pytest.mark.parametrize(
    'embedding_size,cross_num,hidden_size,sparse_feature_num',
    [(8, 0, (32,), 2), (8, 1, (32,), 2)
     ]  # ('auto', 1, (32,), 3) , ('auto', 1, (), 1), ('auto', 1, (32,), 3)
)
def DCNMix_train():
    model_name = "DCN-Mix"
    file = '../dataset/FMRecData'
    HOME_PATH = '/data/app/xxk/DeepCTR'
    test_size = 0.2
    embedding_size = 128
    hidden_units = [256, 128, 64]
    cross_num = 2
    dnn_dropout = 0.5
    feature_columns, train, test, predict = get_train_data(file=file,
                                                           embedding_size=embedding_size,
                                                           test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    np.save("train", train)
    np.save("predict", predict)
    np.save("feature_columns.npy", feature_columns)
    model = DCNMix(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns, cross_num=cross_num, dnn_hidden_units=hidden_units, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, train_X, train_y, test, True, 8)
    np.save("crossnet", model.crossnet)
    np.save("U", model.crossnet.U_list.cpu().detach().numpy())
    np.save("V", model.crossnet.V_list.cpu().detach().numpy())
    np.save("C", model.crossnet.C_list.cpu().detach().numpy())


if __name__ == "__main__":
    DCNMix_train()
