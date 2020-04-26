import pickle as pk

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO
import sys


def f1():
    try:
        file_path = root_dir + 'data/github/test_codevecs_npy/use.codevecs_0.npy'
        with FileIO(file_path, mode="rb") as fio:
            code_vec = np.load(fio)
            print('读取.npy成功', len(code_vec))
            print(code_vec[0][:10])
    except Exception as e:
        print('读取.npy失败')
        print(e)

    try:
        file_path = root_dir + 'data/github/vocab.apiseq.pkl'
        with FileIO(file_path, mode="rb") as fio:
            api_seq_vocab = pk.load(fio)
            print('读取.pkl成功', len(api_seq_vocab))
            print(list(api_seq_vocab.keys())[:3])
    except Exception as e:
        print('读取.pkl失败')
        print(e)

    try:
        file_path = root_dir + 'data/github/use.search.txt'
        with FileIO(file_path, mode="r") as fio:
            lines = fio.readlines()
            print('读取.txt成功')
            print(lines[0])
            print(lines[1])
            print(lines[2])
    except Exception as e:
        print('读取.txt失败')
        print(e)


def f2():
    tf.app.flags.DEFINE_string("tables", "", "tables info")

    FLAGS = tf.app.flags.FLAGS

    print("tables:" + FLAGS.tables)
    tables = [FLAGS.tables]
    # Open a table，打开一个表，返回reader对象
    reader = tf.python_io.TableReader(FLAGS.tables, selected_cols="method_content,method_name,comments")

    # Get total records number, 获得表的总行数
    total_records_num = reader.get_row_count()  # return 3
    print('number of records:', total_records_num)

    batch_size = 2
    for i in range(0, total_records_num, batch_size):
        records = reader.read(batch_size)  # 返回[(25, "Apple", 5.0), (38, "Pear", 4.5)]
        method_content = records[:, 0]
        method_name = records[:, 1]
        comments = records[:, 2]
    # Read records from table, returned by ndarray of records tuples.
    # 读表，返回值将是一个recarray数组，形式为[(uid, name, price)*2]
    # if read again, OutOfRange exception would be thrown，继续读取将抛出OutOfRange异常

    # Close the reader
    reader.close()


def f3():
    a = np.array([5, 4, 3, 2, 1])
    file_path = root_dir + 'a.npy'
    with FileIO(file_path, mode="wb") as fio:
        np.save(fio, a)
    with FileIO(file_path, mode="rb") as fio:
        code_vec = np.load(fio)
    print('读取.npy成功', code_vec)


def f4():
    records = [(b"abc", b"apple", b"5.0"), (b'38', b"pear", b'4.5')]
    records = [[method_content.decode('utf-8'),
                method_name.decode('utf-8'),
                comments.decode('utf-8')] for method_content, method_name, comments in records]
    records = np.array(records)
    print(records[:, 0])


def f5():
    file_path = root_dir + 'java.csv'
    with FileIO(file_path, mode="r") as fio:
        df = pd.read_csv(fio)
    print(df[:3])

if __name__ == '__main__':
    local = sys.platform == 'darwin'
    if local:
        root_dir = '/Users/yilan/Projects/data/'
    else:
        root_dir = 'oss://force-algo/demo/code_recommend/deepcs-clean/'

    f5()
