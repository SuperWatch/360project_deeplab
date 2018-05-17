import tensorflow as tf
temp = tf.tile([1,2,3],[2])
temp2 = tf.tile([[1,2],[3,4],[5,6]],[2,3])
with tf.Session() as sess:
    print(sess.run(temp))
    print(sess.run(temp2))
# [1 2 3 1 2 3]

# [[1 2 1 2 1 2]
# [3 4 3 4 3 4]
# [5 6 5 6 5 6]
# [1 2 1 2 1 2]
# [3 4 3 4 3 4]
# [5 6 5 6 5 6]]