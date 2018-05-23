import tensorflow as tf
import numpy as np

def test_dict():
    direction = {"a" :[1] , "b" : "banana", "c" : "grape", "d" : "orange"}
    print(direction)
    newdict=tf.slice(direction["a"],[0],[3])
    print(newdict)

def test_tile():
    '''tile 像瓦片一样叠加，一样是按照第一维，第二维逐渐深入'''
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

def main():
    test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    out1=np.argmax(test, 0)#输出：[3, 3, 1]
    out2=np.argmax(test, 1)#输出：[2, 2, 0, 0]
    print(out1.dtype)


if __name__ == '__main__':
    main()