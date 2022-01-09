# -*- coding:utf-8 -*-
import time
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
import setting.facenet_args as facenet_args

tf.disable_eager_execution()

class FaceNet:
    """
    facenet
    """
    def __init__(self):
        self.model_exp = facenet_args.model_exp
        self.sess = tf.Session()
        self.__build_net()

    def __build_net(self):
        """
        加载模型建网络
        :return:
        """
        start_time = time.time()
        # 加载模型
        print('Model filename: %s' % self.model_exp)
        with gfile.FastGFile(self.model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        # 获得输入输出tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        model_time = time.time()
        print('\033[32m加载模型时间为{}\033[0m'.format(str(model_time - start_time)))

    def img_to_vetor(self, images):
        """
        将图片转为128维向量
        :param images:
        :return:
        """
        print('\033[32mBegin calculating img vector..\033[0m')
        start_time = time.time()
        # 前向传播计算embeddings
        emb = self.sess.run(
            self.embeddings,
            feed_dict={self.images_placeholder: images, self.phase_train_placeholder: False}
        )
        print('\033[32mFinish calculating img vector, cost time {}..\033[0m'.format(str(time.time() - start_time)))
        return emb
