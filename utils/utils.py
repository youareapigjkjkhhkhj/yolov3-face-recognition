# coding: utf-8
from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

def letterbox_resize(img, new_width, new_height, interp=0):
    """
     Letterbox resize. keep the original aspect ratio in the resized image.
    :param img:
    :param new_width:
    :param new_height:
    :param interp:
    :return:
    """
    ori_height, ori_width = img.shape[:2]

    resize_ratio = min(new_width / ori_width, new_height / ori_height)

    resize_w = int(resize_ratio * ori_width)
    resize_h = int(resize_ratio * ori_height)

    img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)
    image_padded = np.full((new_height, new_width, 3), 128, np.uint8)

    dw = int((new_width - resize_w) / 2)
    dh = int((new_height - resize_h) / 2)

    image_padded[dh: resize_h + dh, dw: resize_w + dw, :] = img

    return image_padded, resize_ratio, dw, dh


def parse_anchors(anchor_path):
    """
    解析anchor文件
    :param anchor_path:
    :return: shape [N, 2], dtype float32
    """
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def read_person_list(map_path):
    """
    读取map映射文件
    :param map_path:
    :return:
    """
    fr = open(map_path, 'r', encoding='utf-8')
    name_list = fr.readline().rstrip().split()
    return name_list


def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    非极大值抑制(tensorflow+gpu)
    :param boxes:
    :param scores:
    :param num_classes:
    :param max_boxes:
    :param score_thresh:
    :param nms_thresh:
    :return:
     params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(
            boxes=filter_boxes, scores=filter_score, max_output_size=max_boxes,
            iou_threshold=nms_thresh, name='nms_indices')

        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def get_color_table(class_num, seed=2):
    """
    多个类别生成不同颜色
    :param class_num: 类别数量
    :param seed:
    :return:
    """
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        rgb = [0, 0, 0]
        while rgb[0] < 100 and rgb[1] < 100 and rgb[2] < 100:  # 不出暗色
            rgb = [random.randint(0, 255) for _ in range(3)]
        color_table[i] = rgb
    return color_table


def get_font(box_width, label):
    """
    获取合适大小的font
    :param box_width:
    :param label:
    :return:
    """
    half_box_w = box_width // 2
    max_size = 20
    min_size = 12
    while min_size <= max_size:
        min_size += 1
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='SimHei')), min_size, encoding="utf-8")
        label_size = font.getsize(label)
        if label_size[0] >= half_box_w:
            return font
    return font


def paint_chinese_opencv(cv2img, chinese_label, font, position, color, fontsize=20):
    """
    cv2 img转PIL输出中文
    :param cv2img:
    :param chinese_label:
    :param font:
    :param position:
    :param color:
    :param fontsize:
    :return:
    """
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    draw.text(position, chinese_label, fill=color, font=font)

    cv2img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return cv2img


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    """
    画出bbox方法
    :param img: 画框图片
    :param coord: [x_min, y_min, x_max, y_max] 格式化坐标
    :param label: 标签名
    :param color: 颜色index
    :param line_thickness: int. 框厚度.
    :return:
    """
    tl = line_thickness or int(round(0.003 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # 太小不写labels
    if label and int(coord[2]) - int(coord[0]) > img.shape[1] // 40:
        # font_size = int(coord[2] - coord[0]) // 27
        # print("font_size:", font_size)
        # font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
        font = get_font(int(coord[2] - coord[0]), label)
        label_size = font.getsize(label)
        c2 = c1[0] + label_size[0], c1[1] - label_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paint_chinese_opencv(cv2img, label, font, (c1[0], c1[1] - label_size[1]), (255, 255, 255))
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img
