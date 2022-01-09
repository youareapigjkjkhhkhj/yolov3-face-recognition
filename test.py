# coding: utf-8

from __future__ import division, print_function
import argparse
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import setting.yolo_args as pred_args
import setting.facenet_args as facenet_args
from utils.utils import gpu_nms, plot_one_box, letterbox_resize
from net.yolo_model import yolov3
from net.facenet_model import FaceNet
# from preprocessing.pre_tools import save_vector_csv, train_face_svm

def build_yolo():
    """
    构建yolo v3网络
    :return:
    """
    with tf.Graph().as_default():
        sess = tf.Session()
        input_data = tf.placeholder(
            tf.float32, [1, pred_args.new_size[1], pred_args.new_size[0], 3], name='input_data'
        )
        with tf.variable_scope('yolov3'):
            yolo_model = yolov3(pred_args.num_class, pred_args.anchors)
            pred_feature_maps = yolo_model.forward(input_data, False)

        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        boxes, scores, labels = gpu_nms(
            pred_boxes, pred_scores, pred_args.num_class,
            max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, pred_args.weight_path)
        return sess, input_data, boxes, scores, labels


def build_facenet():
    """
    构建facenet网络
    :return:
    """
    facenet = FaceNet()
    return facenet


def face_distinguish(facenet, img_ori, boxes_):
    """
    简单使用距离辨别人脸
    :param facenet:
    :param img_ori:
    :param boxes_:
    :return:
    """
    sub_img = []
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        cropped = img_ori[int(y0):int(y1), int(x0):int(x1)]  # 裁剪图片
        cropped = cv2.resize(cropped, (160, 160))
        sub_img.append(cropped)
    img_arr = np.stack(tuple(sub_img))
    vectors = facenet.img_to_vetor(img_arr)  # 得到所有的128维向量

    base_face_vec = pd.read_csv(facenet_args.base_face_csv, index_col=0)
    dis_dic = {}
    for i in range(len(vectors)):
        names = base_face_vec.pop('name')
        dis = np.sqrt(np.square(np.subtract(vectors[i], base_face_vec.values)))
        idx = np.argmin(np.sum(dis, axis=1))
        dis_dic[i] = names[idx]
    return dis_dic


def face_svm_distinguish(facenet, img_ori, boxes_):
    """
    使用svm辨别人脸
    :param facenet:
    :param img_ori:
    :param boxes_:
    :return:
    """
    # 加载svm
    with open(facenet_args.svm_path, 'rb') as in_file:
        (clf, scale_fit) = pickle.load(in_file)
    sub_img = []
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        cropped = img_ori[int(y0):int(y1), int(x0):int(x1)]  # 裁剪图片
        cropped = cv2.resize(cropped, (160, 160))
        sub_img.append(cropped)
    img_arr = np.stack(tuple(sub_img))
    vectors = facenet.img_to_vetor(img_arr)  # 得到所有的128维向量
    # 标准化
    vectors = scale_fit.transform(vectors)
    labels = clf.predict(vectors)
    print("person labels", labels)
    name_labels = [facenet_args.person_list[i] for i in labels]
    return name_labels, labels


def img_detect(input_args):
    """
    人脸辨别
    :param input_args:
    :return:
    """
    sess, input_data, boxes, scores, labels = build_yolo()
    facenet = build_facenet()

    img_ori = cv2.imread(input_args.input_image)  # opencv 载入
    if pred_args.use_letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, pred_args.new_size[0], pred_args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(pred_args.new_size))

    # img 转RGB, 转float, 归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # 还原坐标到原图
    if pred_args.use_letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori / float(pred_args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori / float(pred_args.new_size[1]))

    print('box coords:', boxes_, '\n' + '*' * 30)
    print('scores:', scores_, '\n' + '*' * 30)
    print('labels:', labels_)

    labels, labels_idx = face_svm_distinguish(facenet, img_ori, boxes_)
    # 遍历每一个bbox
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]

        if labels != '':
            img_ori = plot_one_box(
                img_ori, [x0, y0, x1, y1],
                label=labels[i],
                color=facenet_args.color_table[labels_idx[i]]
            )
    cv2.imwrite(pred_args.output_image, img_ori)
    cv2.imshow('Detection result', img_ori)
    cv2.waitKey(0)
    sess.close()


def video_detect(input_args):
    sess, input_data, boxes, scores, labels = build_yolo()
    facenet = build_facenet()
    vid = cv2.VideoCapture(input_args.input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(pred_args.output_video, fourcc, video_fps, (video_width, video_height))

    for i in range(video_frame_cnt):
        ret, img_ori = vid.read()
        if pred_args.use_letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, pred_args.new_size[0], pred_args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(pred_args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        # 还原坐标到原图
        if pred_args.use_letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(pred_args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(pred_args.new_size[1]))

        print('box coords:', boxes_, '\n' + '*' * 30)
        print('scores:', scores_, '\n' + '*' * 30)
        print('labels:', labels_)

        labels, labels_idx = face_svm_distinguish(facenet, img_ori, boxes_)
        end_time = time.time()

        # 遍历每一个bbox
        for j in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[j]

            if labels != '':
                img_ori = plot_one_box(
                    img_ori, [x0, y0, x1, y1],
                    label=labels[j],
                    color=facenet_args.color_table[labels_idx[j]]
                )

        cv2.putText(
            img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000),
            (40, 40), 0, fontScale=1, color=(0, 255, 0), thickness=2
        )
        cv2.imshow('Detection result', img_ori)
        video_writer.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    video_writer.release()


def main():
    parser = argparse.ArgumentParser(description='YOLO V3 检测文件')
    parser.add_argument('--detect_object', default=pred_args.detect_object, type=str, help='检测目标-img或video')
    parser.add_argument('--input_image', default=pred_args.input_image, type=str, help='图片路径')
    parser.add_argument('--input_video', default=pred_args.input_video, type=str, help='视频路径')

    input_args = parser.parse_args()
    # 图片检测
    if input_args.detect_object == 'img':
        img_origin = cv2.imread(input_args.input_image)  # 原始图片
        if img_origin is None:
            raise Exception('未找到图片文件！')
        img_detect(input_args)

    # 视频检测
    elif input_args.detect_object == 'video':
        vid = cv2.VideoCapture(input_args.input_video)
        if vid is None:
            raise Exception('未找到视频文件!')
        video_detect(input_args)


if __name__ == '__main__':
    """
    创建新数据集后，仅第一次需运行 save_vector_csv、 train_face_svm 函数，之后注释即可 (16行引用同理)
    """
    # save_vector_csv()
    # train_face_svm()

    main()
