# -*- coding:utf-8 -*-

import os
import cv2
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from net.facenet_model import FaceNet
import setting.facenet_args as facenet_args
from root_path import project_dir

"""
提取base face文件, 保存为128维向量的csv文件
"""


def convert_base_face_to_vector(facenet):
    """
    获得文件转成128维向量
    :param facenet:
    :return:
    """
    vector_list = []
    for dir in os.listdir(facenet_args.base_face_dir):
        real_dir = os.path.join(facenet_args.base_face_dir, dir)  # 单人的文件夹
        if os.path.isdir(real_dir):
            tag = dir  # 人名
            img_list = []
            for file in os.listdir(real_dir):  # 每张图片
                file_path = os.path.join(real_dir, file)
                img_origin = cv2.imread(file_path)
                img_160 = cv2.resize(img_origin, (160, 160))
                img_list.append(img_160)
            img_arr = np.stack(tuple(img_list))  # 拼接图片arr, shape=?*160*160*3

            vector = facenet.img_to_vetor(img_arr)  # 某人所有图片的128维向量
            for i in range(vector.shape[0]):
                vec_list = vector[i].tolist()
                vec_list.insert(0, tag)
                vector_list.append(vec_list)
    return pd.DataFrame(vector_list)


def save_vector_csv():
    """
    将图片转为128维向量并储存到csv
    :return:
    """
    head = list(range(512))
    head.insert(0, 'name')
    facenet = FaceNet()
    vector_df = convert_base_face_to_vector(facenet)
    vector_df.to_csv(facenet_args.base_face_csv, header=head)


def train_face_svm():
    """
    根据储存的128维向量训练一个svm分类器
    :return:
    """
    # 读取存好的128维向量
    data = pd.read_csv(facenet_args.base_face_csv, index_col=0)
    names = data.pop('name')
    x = data.values

    # 标准化
    scale = StandardScaler()
    scale_fit = scale.fit(x)
    x = scale_fit.transform(x)
    y = names.values

    # svm
    clf = SVC(probability=True, kernel='linear')
    clf.fit(x, y)

    # 储存模型和归一化参数书
    # joblib.dump(clf, facenet_args.svm_path)
    with open(facenet_args.svm_path, 'wb') as outfile:
        pickle.dump((clf, scale_fit), outfile)


def test_svm():
    """

    :return:
    """
    data = pd.read_csv(facenet_args.base_face_csv, index_col=0)
    data.pop('name')
    # clf2 = joblib.load(facenet_args.svm_path)
    with open(facenet_args.svm_path, 'rb') as in_file:
        (clf2, scale_fit) = pickle.load(in_file)
    # 测试读取后的Model
    for i in range(data.shape[0]):
        vec = np.array(data.iloc[i])
        vec_2d = scale_fit.transform([vec])
        # print(clf.predict([vec]))
        predictions = clf2.predict_proba(vec_2d)
        print(predictions)
        best_class_indices = np.argmax(predictions, axis=1)
        print(best_class_indices)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        print(best_class_probabilities)


if __name__ == '__main__':
    # 将图片转为128为向量并储存
    save_vector_csv()

    # 用储存的向量训练一个svm分类器
    train_face_svm()

    # 测试
    test_svm()
