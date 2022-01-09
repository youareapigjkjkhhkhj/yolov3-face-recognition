# coding=utf-8
from __future__ import division, print_function
import tensorflow.compat.v1 as tf
import tf_slim as slim
from net.net_module import darknet53, detect_net


class yolov3(object):
    """
    Yolo v3
    """
    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999, weight_decay=5e-4, use_static_shape=True):
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                    biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                    weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53(inputs)

                with tf.variable_scope('yolov3_head'):
                    feature_map_1, feature_map_2, feature_map_3 = detect_net(
                        route_1, route_2, route_3, self.use_static_shape, self.class_num
                    )

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        """
        转移层，特征融合(Fine-Grained Features)，把高分辨率的浅层特征连接到低分辨率的生成特征，堆积在不同channel上。
        :param feature_map_i: 不同尺度的 feature map，
        :param anchors: 3个anchor，shape=3,2
        :return:
        """
        # 选用tf.shape()和tensor.get_shape(),得到grid划分(前者快一点点)
        # 格式为[h, w], 格子划分13*13，,26*26, 52*52
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]  # [13, 13]
        # 每个cell的大小，转成float32, [w, h]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # 转换anchor数据符合feature map, 注意顺序
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        # 3个feature map channel都是255.
        # 3*(1+4+4*20)=255, 含义为3个anchor boxes, 4个pre_boxes, 1个置信度confidence, 和80个类别的概率
        # 将feture(1,13,13,255)转为shape=[?, grid_h, grid_w, 3, (5+80)]=(1,13,13,3,85)
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # 将其在最后一维分割成4个tensor, 2个[?, grid_h, grid_w, 3, 2], [?, grid_h, grid_w, 3, 1], [?, grid_h, grid_w, 3, 20]
        # 分别是(1,13,13,3,2)(1,13,13,3,2)(1,13,13,3,1)(1,13,13,3,80)
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)

        # logistic预测的, 所以sigmoid激活
        box_centers = tf.nn.sigmoid(box_centers)

        # 通过一些广播技巧，获得网格的坐标
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # 获得框在feature map上的绝对坐标, 转换为原图坐标
        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        # tf.clip_by_value避免nan值
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # 转换成初始图片尺度
        box_sizes = box_sizes * ratio[::-1]

        # [N, 13, 13, 3, 4]
        # 最后的维度: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4],  转换成初始图片尺度
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        """
        提取的Feature map进行后续步骤，转化为bboxes信息
        :return:
        """

        def _reshape(result):
            """
            每种尺度改变形状
            :param result:
            :return:
            """
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        print("\033[32mBegin building feature map to bboxes op...")
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [
            (feature_map_1, self.anchors[6:9]),  # (116,90), (156,198), (373,326)
            (feature_map_2, self.anchors[3:6]),  # (30,61), (62,45), (59,119),
            (feature_map_3, self.anchors[0:3])  # (10,13), (16,30), (33,23),
        ]  # 针对不同grid使用不同大小的anchor

        # 对每种尺度进行特征融合
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # 三种尺度的结果, 416*416 举例
        # [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        print("\033[32mFinish building feature map to bboxes op...")
        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        """
        计算损失函数
        :param feature_map_i:  feature maps [N, 13, 13, 3*(5 + num_class)]
        :param y_true: y_ture [N, 13, 13, 3, 5 + num_class + 1]
        :param anchors: [3, 2]
        :return:
        """
        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def box_iou(self, pred_boxes, valid_true_boxes):
        """
        计算交并比
        :param pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        :param valid_true_boxes: [V, 4]
        :return:
        """
        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    def compute_loss(self, y_pred, y_true):
        """
        计算损失
        :return:
        """
        print("\033[32mBegin building compute loss op...")
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # 计算3种维度的5种损失
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        print("\033[32mFinish building compute loss op...")
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]
