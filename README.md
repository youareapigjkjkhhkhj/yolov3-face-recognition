> 转载自：https://gitee.com/windandwine/Argus
>
> 做部分改动使其可在 TensorFlow2.0 下运行

# 权重下载

## FaceNet

下载[20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)模型

将`20180402-114759.pb`放置`weights_facenet2`文件夹下。

## YOLOv3

🔗[下载链接，密码：qli4](https://pan.baidu.com/s/1ifCaB2ASPQFPN1XGPGyplg)

将解压后文件夹内文件移动置`weights_yolo`文件夹下

## 文件权重位置

文件权重位置如下图所示：

<center>
    <img src="https://github.com/laugh12321/yolov3-face-recognition/blob/main/data/file_tree.png" />
</center>

# 使用方法

创建虚拟环境（推荐`python 3.8.12`）

```
conda create -n your_env_name python=3.8.12
```

激活虚拟环境并安装包

```
activate your_env_name

conda install --yes --file requirements.txt
```

运行代码

```
python test.py
```

<b>注1: </b>创建新数据集后，创建新数据集后，仅第一次需运行`test.py`第248、249行的 `save_vector_csv`、 t`rain_face_svm` 函数，之后注释即可 (16行引用同理)

<b>注2: </b>具体使用方法参考👉[原仓库](https://gitee.com/windandwine/Argus/blob/master/README.md)