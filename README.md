> 项目地址：https://gitee.com/windandwine/Argus
> 转载请注明出处

# 一、项目简介

fine-tune YOLO v3 + FaceNet进行人脸识别，辨别。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0917/164808_5e279300_1295352.png "result.png")

## 1. 项目结构

--data  

--|------baseface 图片、根据这些图片训练的128d向量，以及文件夹与人名的映射文件

---------|------0  第一个人的图片tag=0

---------|------1  第二个人的图片tag=1

......

---------|------n  第n个人的图片tag=n

---------|------map.txt  文件夹与人名的映射，依次放即可

---------|------vector.csv 根据这些图片得到的128维向量以及其类别（文件夹名）

--|------weights_facenet     模型文件（facenet）

--|------weights_yolo  模型（fune-tuning的yolo v3）

--|------weights_svm 模型（根据csv文件训练的svm模型）

--|------face-names 预测类别，默认即可

--|------yolo_anchors.txt  训练yolo v3时聚类得到的anchors框

--net  yolo和facenet的网络

--preprocessing  预处理工具，下文的使用方法前期步骤都需要在这里运行

--setting  模型参数，可根据需要修改

--utils 封装的一些方法

--test.py 主体方法


## 2.模型文件（权重）
yolo_face模型

yolo v3是基于论文作者模型，在Wider Face数据上fine-tuning的。
可以到[这里](https://pan.baidu.com/s/1ifCaB2ASPQFPN1XGPGyplg)下载，密码**qli4**。

facenet模型

1. model-20170512-110547，请到[这里](https://pan.baidu.com/s/1pGDUIODBZ2Rvrr1MTRl9ZQ)下载，密码**9x2l**
2. model-20180408-102900，请到[这里](https://pan.baidu.com/s/1xe2dTQsgaXf3xTmCfVprDQ)下载，密码**jjf1**
3. model-20180402-114759（**推荐**），请到[这里](https://pan.baidu.com/s/18dOokPTF3JXFix99aUfz6Q)下载，密码**56mh**


## 3.yolo v3

YOLO v3的详细预测和训练，可到本人另一个项目[YOLO_v3_tensorflow](https://gitee.com/windandwine/YOLO_v3_tensorflow)了解。

## 4.踩的坑

1. fine-tune yolo v3时使用的是wider face数据集，其中有两个标注框是宽度或高度为0的，错误标注，筛选的时候需要去除掉这两个标注框，否则nms会报除0的异常。
2. 训练svm时，需要标准化，预测时需要用同参数标准化再预测，否则svm预测结果都相同。
3. 之前使用model-20170512-110547模型，输出128d向量，效果不好，换成model-20180402-114759模型，输出512d向量，效果有所提升。
4. 每个人15张脸部图片，训练svm效果一般，可以增大样本量或者使用一些svm的tricks。

# 三、使用方法

项目需要安装以下包

```
numpy==1.16.4
pandas==0.24.2
opencv-python==4.1.1
scikit-learn=0.21.2
tensorflow==1.13.1
pillow==6.1.0
```

## 1.放置模型文件

下载yolo_face和facenet模型文件，放置在**data**文件夹下，如图

![输入图片说明](https://images.gitee.com/uploads/images/2019/0927/191846_ea9a5d31_1295352.png "weights_here.png")

## 2.制作自己的人脸数据集

截取需要识别的人物的脸部图片，一人一个文件夹，文件夹名称从0开始依次累加，放在路径**data/base_face**下。

![输入图片说明](https://images.gitee.com/uploads/images/2019/0917/111219_bbb5330d_1295352.png "face.png")

并修改**data/map.txt**，以空格分隔，下标和人脸文件夹名一一对应（map.txt中第一条下标为0，对应data/base_face/0文件夹）。

## 3.使用工具将图片转换成向量并存储

运行**preprecessing/pre_tools.py**内的**save_vector_csv()**，自动将图片使用facenet转换为128d或512d向量，并存储为**data/base_face/vector.csv**。

## 4.训练svm分类器

基于已经储存的vector.csv文件，进行**标准化**后，运行**preprecessing/pre_tools.py**内的**train_face_svm()**，使用scikit-learn训练svm模型，并储存在**data/weights_svm/svm.pkl**中。

## 5.开始测试

### 5.1图片测试

放在**data/test_img**下，将**setting/yolo_args.py**中的**detect_object**改为**img**，**input_image** 改为图片路径。

运行根目录下的**test.py**文件。

另外如果要存储检测后的图片，将**setting/yolo_args.py**中的**output_image**改为要存储的路径，并**保证各层级文件夹存在**。

### 5.2视频测试

放在**data/test_video**下，将**setting/yolo_args.py**中的**detect_object**改为**video**，**input_video** 改为视频路径。

运行根目录下的**test.py**文件。

另外如果要存储检测后的视频，将**setting/yolo_args.py**中的**output_video**改为要存储的路径，并**保证各层级文件夹存在**。

如有疑问，请贴在issues。