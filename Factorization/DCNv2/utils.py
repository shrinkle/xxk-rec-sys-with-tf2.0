# -*- coding: utf-8 -*-
import os

import numpy as np
import torch as torch

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from collections import Counter

SAMPLE_SIZE = 64
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1,
                                                                                                         sample_size)


def get_test_data(sample_size=1000, embedding_size=4, sparse_feature_num=1, dense_feature_num=1,
                  sequence_feature=['sum', 'mean', 'max'], classification=True, include_length=False,
                  hash_flag=False, prefix=''):


    feature_columns = []
    model_input = {}


    if 'weight'  in sequence_feature:
        feature_columns.append(VarLenSparseFeat(SparseFeat(prefix+"weighted_seq",vocabulary_size=2,embedding_dim=embedding_size),
                                                maxlen=3,length_name=prefix+"weighted_seq"+"_seq_length",weight_name=prefix+"weight"))
        s_input, s_len_input = gen_sequence(
            2, 3, sample_size)

        model_input[prefix+"weighted_seq"] = s_input
        model_input[prefix+'weight'] = np.random.randn(sample_size,3,1)
        model_input[prefix+"weighted_seq"+"_seq_length"] = s_len_input
        sequence_feature.pop(sequence_feature.index('weight'))


    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(prefix+'sparse_feature_'+str(i), dim,embedding_size,dtype=torch.int32))
    for i in range(dense_feature_num):
        feature_columns.append(DenseFeat(prefix+'dense_feature_'+str(i), 1,dtype=torch.float32))
    for i, mode in enumerate(sequence_feature):
        dim = np.random.randint(1, 10)
        maxlen = np.random.randint(1, 10)
        feature_columns.append(
            VarLenSparseFeat(SparseFeat(prefix +'sequence_' + mode,vocabulary_size=dim,  embedding_dim=embedding_size), maxlen=maxlen, combiner=mode))

    for fc in feature_columns:
        if isinstance(fc,SparseFeat):
            model_input[fc.name]= np.random.randint(0, fc.vocabulary_size, sample_size)
        elif isinstance(fc,DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
        else:
            s_input, s_len_input = gen_sequence(
                fc.vocabulary_size, fc.maxlen, sample_size)
            model_input[fc.name] = s_input
            if include_length:
                fc.length_name = prefix+"sequence_"+str(i)+'_seq_length'
                model_input[prefix+"sequence_"+str(i)+'_seq_length'] = s_len_input

    if classification:
        y = np.random.randint(0, 2, sample_size)
    else:
        y = np.random.random(sample_size)

    return model_input, y, feature_columns


def get_train_data(file, embedding_size=4, test_size=0.2):

    names = ['label', 'uid', 'vid', 'u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1',
             'u_m_cate2', 'u_m_cate3', 'u_m_tag', 'u_m_kis', 'v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag',
             'v_priority', 'v_author', 'v_producerType', 'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime',
             'v_playcount', 'v_timeLen']

    sample_num = 20000000
    data_df = pd.read_csv(file, sep=',', iterator=True, header=None, names=names)
    data_df = data_df.get_chunk(sample_num)

    sparse_features = ['u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1', 'u_m_cate2',
                       'u_m_cate3', 'u_m_tag', 'u_m_kis',
                       'v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author', 'v_producerType',
                       'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime']
    dense_features = ['v_playcount', 'v_timeLen']

    sparse_feature_num = len(sparse_features)
    dense_feature_num = len(dense_features)

    predict_uid_feat = ['u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1', 'u_m_cate2',
                        'u_m_cate3', 'u_m_tag', 'u_m_kis']

    predict_uid2vid_feat = ['v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author',
                            'v_producerType', 'uid']
    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])
    dense_features = [feat for feat in data_df.columns if feat not in sparse_features + ['label', 'vid', 'uid']]
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])

    feature_columns = [DenseFeat('dense_feature_'+feat, 1, dtype=torch.float32) for feat in dense_features] + \
                      [SparseFeat('sparse_feature_'+feat, len(data_df[feat].unique()),embedding_size,dtype=torch.int32)
                       for feat in sparse_features]
    np.save("data_df", data_df)
    np.save("feature_columns", feature_columns)
    train, test = train_test_split(data_df, test_size=test_size, shuffle=True, stratify=data_df['label'])

    train_x = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_x = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
    test_y = test['label'].values.astype('int32')
    data_uid = data_df.drop_duplicates(subset='uid', keep='first')
    predict_uid = [data_uid[['uid']].values.astype('str'),
                   data_uid[predict_uid_feat].values.astype('int32')]
    data_vid = data_df.drop_duplicates(subset='vid', keep='first')
    sparse_features_predict = ['v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author', 'v_producerType',
                               'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime']
    predict_vid = [data_vid[['vid']].values.astype('str'),
                   data_vid[dense_features].values.astype('float32'),
                   data_vid[sparse_features_predict].values.astype('int32')]

    data_uid2vid = data_df[predict_uid2vid_feat].groupby("uid")
    data_uid2vid_feat_list = []
    for (uid, info) in list(data_uid2vid):
        counter = Counter()
        counter.update(np.reshape(np.array(info), -1))
        count = counter.most_common(10)
        uid_feat = [fea for (fea, num) in count if fea != 0 and fea != uid]
        data_uid2vid_feat_list.append(uid_feat)

    predict_uid2vid = [data_uid[['uid']].values.astype('str'), data_uid2vid_feat_list]

    return feature_columns, (train_x, train_y), (test_x, test_y), (predict_uid, predict_vid, predict_uid2vid)



def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def get_train_data_criteo(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    names = ['label', 'uid', 'vid', 'u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1',
             'u_m_cate2', 'u_m_cate3', 'u_m_tag', 'u_m_kis', 'v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag',
             'v_priority', 'v_author', 'v_producerType', 'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime',
             'v_playcount', 'v_timeLen']

    if read_part:
        data_df = pd.read_csv(file, sep=',', iterator=True, header=None, names=names)
        data_df = data_df.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file, sep=',', header=None, names=names)

    sparse_features = ['u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1', 'u_m_cate2',
                       'u_m_cate3', 'u_m_tag', 'u_m_kis',
                       'v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author', 'v_producerType',
                       'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime']
    dense_features = ['v_playcount', 'v_timeLen']

    predict_uid_feat = ['u_w_cate1', 'u_w_cate2', 'u_w_cate3', 'u_w_tag', 'u_w_kis', 'u_m_cate1', 'u_m_cate2',
                        'u_m_cate3', 'u_m_tag', 'u_m_kis']

    predict_uid2vid_feat = ['v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author',
                            'v_producerType', 'uid']
    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])
    dense_features = [feat for feat in data_df.columns if feat not in sparse_features + ['label', 'vid', 'uid']]
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])

    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
                        for feat in sparse_features]]
    np.save("data_df", data_df)
    np.save("feature_columns", feature_columns)

    train, test = train_test_split(data_df, test_size=test_size)

    train_x = [train[dense_features].values.astype('float32'), train[sparse_features].values.astype('int32')]
    train_y = train['label'].values.astype('int32')
    test_x = [test[dense_features].values.astype('float32'), test[sparse_features].values.astype('int32')]
    test_y = test['label'].values.astype('int32')
    data_uid = data_df.drop_duplicates(subset='uid', keep='first')
    predict_uid = [data_uid[['uid']].values.astype('str'),
                   data_uid[predict_uid_feat].values.astype('int32')]
    data_vid = data_df.drop_duplicates(subset='vid', keep='first')
    sparse_features_predict = ['v_fc', 'v_sc', 'v_tc', 'v_kis', 'v_album', 'v_tag', 'v_priority', 'v_author', 'v_producerType',
                               'v_tags', 'v_extags', 'v_loc', 'v_isper', 'v_uptime']
    predict_vid = [data_vid[['vid']].values.astype('str'),
                   data_vid[dense_features].values.astype('float32'),
                   data_vid[sparse_features_predict].values.astype('int32')]

    data_uid2vid = data_df[predict_uid2vid_feat].groupby("uid")
    data_uid2vid_feat_list = []
    for (uid, info) in list(data_uid2vid):
        counter = Counter()
        counter.update(np.reshape(np.array(info), -1))
        count = counter.most_common(10)
        uid_feat = [fea for (fea, num) in count if fea != 0 and fea != uid]
        data_uid2vid_feat_list.append(uid_feat)

    predict_uid2vid = [data_uid[['uid']].values.astype('str'), data_uid2vid_feat_list]

    return feature_columns, (train_x, train_y), (test_x, test_y), (predict_uid, predict_vid, predict_uid2vid)

def layer_test(layer_cls, kwargs = {}, input_shape=None, 
               input_dtype=torch.float32, input_data=None, expected_output=None,
               expected_output_shape=None, expected_output_dtype=None, fixed_batch_size=False):
    '''check layer is valid or not

    :param layer_cls:
    :param input_shape:
    :param input_dtype:
    :param input_data:
    :param expected_output:
    :param expected_output_dtype:
    :param fixed_batch_size:

    :return: output of the layer
    '''
    if input_data is None:
        # generate input data
        if not input_shape:
            raise ValueError("input shape should not be none")

        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        
        if all(isinstance(e, tuple) for e in input_data_shape):
            input_data = []
            for e in input_data_shape:
                rand_input = (10 * np.random.random(e))
                input_data.append(rand_input)
        else:
            rand_input = 10 * np.random.random(input_data_shape)
            input_data = rand_input

    else:
        # use input_data to update other parameters
        if input_shape is None:
            input_shape = input_data.shape
    
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype
    
    # layer initialization
    layer = layer_cls(**kwargs)
    
    if fixed_batch_size:
        inputs = torch.tensor(input_data.unsqueeze(0), dtype=input_dtype)
    else:
        inputs = torch.tensor(input_data, dtype=input_dtype)
    
    # calculate layer's output
    output = layer(inputs)

    if not output.dtype == expected_output_dtype:
        raise AssertionError("layer output dtype does not match with the expected one")
    
    if not expected_output_shape:
            raise ValueError("expected output shape should not be none")

    actual_output_shape = output.shape
    for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
        if expected_dim is not None:
            if not expected_dim == actual_dim:
                raise AssertionError(f"expected_dim:{expected_dim}, actual_dim:{actual_dim}")
    
    if expected_output is not None:
        # check whether output equals to expected output
        assert_allclose(output, expected_output, rtol=1e-3)
    
    return output


def check_model(model, model_name, train_x, train_y, test, check_model_io=True, epochs=1):
    '''
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y:
    :param check_model_io:
    :return:
    '''

    model.compile('adam', 'binary_crossentropy',
    #              metrics=['binary_crossentropy', 'acc', 'auc', 'mse'])
                  metrics = ['binary_crossentropy', 'auc', 'mse'])
    model.fit(train_x, train_y, batch_size=100, epochs=epochs, validation_data=test)  # validation_split=0.5)
    auc = model.evaluate(test[0], test[1])
    print('test AUC: ', auc)

    print(model_name + 'test, train valid pass!')
    torch.save(model.state_dict(), model_name + '_weights.h5')
    model.load_state_dict(torch.load(model_name + '_weights.h5'))
    # os.remove(model_name + '_weights.h5')
    print(model_name + 'test save load weight pass!')
    if check_model_io:
        torch.save(model, model_name + '.h5')
        #model = torch.load(model_name + '.h5')
        # os.remove(model_name + '.h5')
        print(model_name + 'test save load model pass!')
    print(model_name + 'test pass!')

def get_device(use_cuda = True):
    device = 'cpu'
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    return device
