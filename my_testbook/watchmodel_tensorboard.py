import tensorflow as tf

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(
    tf.gfile.FastGFile(
        "F:\\Modellib\\frozen_inference_graph_mobilenetv2.pb",
        # "F:\\Modellib\\deeplabv3_mnv2_pascal_train_aug\\frozen_inference_graph.pb",
        'rb').read())
#如果不用‘rb’或者使用‘b’格式导入，会有utf-8错误。
_ = tf.import_graph_def(graphdef, name="")

# _=tf.train.import_meta_graph("C:\\Users\\v-zhaojinqiang-os\\trans\\PNet_landmark\\PNet-30.meta")
writer = tf.summary.FileWriter("F:\\Loglib\\log2", graph)
#如果使用graphdef会有警告，graphdef是类似tf的关键字
# writer.close()
# 然后打开anaconda promote ，输入tensorboard --logdir “所在地址”
# 然后打开anaconda promote ，输入tensorboard --logdir F:/Loglib/log
# 最后这句命令是控制台命令，寻址符号要反过来/,而不是\
