# -*- coding:utf-8 -*-
from utils.utils import read_person_list, get_color_table
from root_path import project_dir

model_exp = project_dir + '/data/weights_facenet2/20180402-114759.pb'
base_face_dir = project_dir + '/data/base_face'  # 基图片文件夹
base_face_csv = project_dir + '/data/base_face/vector.csv'  # 基图片转成128为nparray的npz文件
svm_path = project_dir + '/data/weights_svm/svm.pkl'  # 分类器模型文件
map_path = project_dir + '/data/base_face/map.txt'

person_list = read_person_list(map_path)
person_num = len(person_list)

color_table = get_color_table(person_num)  # 根据类别数生成颜色列表
