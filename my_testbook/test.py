# import tensorflow as tf

# graph = tf.get_default_graph()
# graphdef = graph.as_graph_def()
# graphdef.ParseFromString(
#     tf.gfile.FastGFile(
#         "F:\\Modellib\\deeplabv3_mnv2_pascal_train_aug\\frozen_inference_graph.pb",
#         'rb').read())
# #如果不用‘rb’或者使用‘b’格式导入，会有utf-8错误。
# _ = tf.import_graph_def(graphdef, name="")

# # _=tf.train.import_meta_graph("C:\\Users\\v-zhaojinqiang-os\\trans\\PNet_landmark\\PNet-30.meta")
# writer = tf.summary.FileWriter("F:\\VScodeSusu\\testcode\\log", graph)
# #如果使用graphdef会有警告，graphdef是类似tf的关键字
# # writer.close()
# # 然后打开anaconda promote ，输入tensorboard --logdir F:\VScodeSusu\testcode\log

###
###
# import tensorflow as tf
# temp = tf.tile([1,2,3],[2])
# temp2 = tf.tile([[1,2],[3,4],[5,6]],[2,3])
# with tf.Session() as sess:
#     print(sess.run(temp))
#     print(sess.run(temp2))

# # [1 2 3 1 2 3]

# # [[1 2 1 2 1 2]
# # [3 4 3 4 3 4]
# # [5 6 5 6 5 6]
# # [1 2 1 2 1 2]
# # [3 4 3 4 3 4]
# # [5 6 5 6 5 6]]