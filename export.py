from absl import app
from absl import flags
from absl import logging
import numpy as np
import os.path as path
from sklearn.model_selection import train_test_split

import transformations.reader.generic as generic_reader
import transformations.tfhub_module as tfhub_module
import transformations.torchhub_model as torchhub_model
import transformations.pca as pca
import transformations.nca as nca
import transformations.random_proj as random_proj

FLAGS = flags.FLAGS

# 数据来源
flags.DEFINE_enum("variant", None, ["matrix", "textfile", "folder", "mnist_data", "cifar_data", "tfds", "torchvision"], "Input for running the tool")
# 按顺序要应用的变换名列表
# 为空则导出原始特征
flags.DEFINE_list("transformations", [], "List of transformations (can be empty, which exports the raw features) to be applied in that order (starting from the second only using a matrix as a input")
# 可选的子采样数量
flags.DEFINE_integer("subsamples", None, "Number of subsamples to export")
# 特征和标签矩阵的导出路径
flags.DEFINE_string("export_path", ".", "Path to folder (should exist) where the features and labels matrices should be stored")
# 特征的导出文件名
flags.DEFINE_string("export_features", None, "Features export file name")
# 标签的导出文件名
flags.DEFINE_string("export_labels", None, "Labels export file name")

def _get_transform_fns():
    fns = []
    for t in FLAGS.transformations:
        # strip()去掉字符串两端的空白字符，lower()将字符串转换为小写
        val = t.strip().lower()
        if val == "tfhub_module":
            # len(fns) == 0表示这是第一个变换函数
            fns.append(tfhub_module.load_and_apply if len(fns) == 0 else tfhub_module.apply)
        elif val == "pca":
            fns.append(pca.load_and_apply if len(fns) == 0 else pca.apply)
        elif val == "nca":
            fns.append(nca.load_and_apply if len(fns) == 0 else nca.apply)
        elif val == "random_proj":
            fns.append(random_proj.load_and_apply if len(fns) == 0 else random_proj.apply)
        elif val == "torchhub_model":
            fns.append(torchhub_model.load_and_apply if len(fns) == 0 else torchhub_model.apply)
        else:
            raise app.UsageError("Transformation '{}' is not valid!".format(t))
    
    return fns


def main(argv):
    # 确认导出路径存在
    if not path.exists(FLAGS.export_path):
        raise app.UsageError("Path to the export folder '{}' needs to exist!".format(FLAGS.export_path))
    
    # 如果数据来源是矩阵且没有变换，报错
    if FLAGS.variant == "matrix" and len(FLAGS.transformations) == 0:
        raise app.UsageError("Loading and rexporting the labels and features matrix without transformation is stupid! Use the command line and 'cp'!")

    # Apply transformations
    transform_fns = _get_transform_fns()
    for i, fn in enumerate(transform_fns):
        if i == 0:
            features, dim, samples, labels = fn()
        else:
            features, dim, samples, labels = fn(features, dim, samples, labels)

    if len(transform_fns) == 0:
        features, dim, samples, labels = generic_reader.read()

    if FLAGS.subsamples is not None and FLAGS.subsamples > 0 and FLAGS.subsamples < samples:
        logging.log(logging.INFO, "Subsampling {} sampels".format(FLAGS.subsamples))
        features, _, labels, _ = train_test_split(features,
                                                  labels,
                                                  test_size = None,
                                                  train_size = FLAGS.subsamples,
                                                  stratify = labels)
        samples = FLAGS.subsamples

    # Export data
    export_folder = FLAGS.export_path
    features_path = path.join(export_folder, FLAGS.export_features)
    logging.log(logging.INFO, "Saving features with shape {} to '{}'".format(np.shape(features), features_path))
    np.save(features_path, features)

    labels_path = path.join(export_folder, FLAGS.export_labels)
    logging.log(logging.INFO, "Saving labels with shape {} to '{}'".format(np.shape(labels), labels_path))
    np.save(labels_path, labels)


if __name__ == "__main__":
    # flags.mark_flag_as_required的输入是一个flag名称，名称是一个字符串，无返回值
    # 作用是把指定的flag标记为必需的，如果在运行程序时没有提供该flag的值，程序会报错并提示用户提供该flag
    flags.mark_flag_as_required("variant")
    flags.mark_flag_as_required("export_features")
    flags.mark_flag_as_required("export_labels")
    app.run(main)
