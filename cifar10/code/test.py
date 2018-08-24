# Source: CS342 with Dr. Krähenbühl
import tensorflow as tf
from glob import glob
import numpy as np
import cv2
import pandas as pd

def convert_images(files):
    return np.asarray([cv2.imread(file) for file in files])

def load(input_file, graph=None, session=None):
    from os import path
    from shutil import rmtree
    from tempfile import mkdtemp
    from zipfile import ZipFile

    tmp_dir = mkdtemp()

    f = ZipFile(input_file, 'r')
    f.extractall(tmp_dir)
    f.close()

    # Find the model name
    meta_files = glob(path.join(tmp_dir, '*.meta'))
    if len(meta_files) < 1:
        raise IOError( "[E] No meta file found, giving up" )
    if len(meta_files) > 1:
        raise IOError( "[E] More than one meta file found, giving up" )

    meta_file = meta_files[0]
    model_file = meta_file.replace('.meta', '')

    if graph is None:
        graph = tf.get_default_graph()
    if session is None:
        session = tf.get_default_session()
    if session is None:
        session = tf.Session()

    # Load the model in TF
    with graph.as_default():
        saver = tf.train.import_meta_graph(meta_file)
        if saver is not None:
            saver.restore(session, model_file)
    rmtree(tmp_dir)
    return graph

if __name__ == '__main__':
    sess = tf.Session()
    graph = load('resnet32_wide2.tfg',session=sess)
    test_files = glob('./test/*') # expects test images to be in test directory
    test_files = sorted(test_files, key = lambda x:int(x.split('\\')[1].split('.png')[0])) # need to change it to work for any system, only Windows right now
    
    #image_test = convert_images(test_files)
    image_test = np.load('test.npy')
    print("Loaded Data!")
    labels_df = pd.read_csv('trainLabels.csv')
    labels_unique = labels_df.label.unique()

    with graph.as_default():
        test_batch = 1000 # batch shape
        predictions = []

        output = graph.get_tensor_by_name('output:0')
        final_input = graph.get_tensor_by_name('Identity:0')
        is_training = graph.get_tensor_by_name('Placeholder_2:0')

        for i in range(0,image_test.shape[0],test_batch):
            if i%90000==0:
                print(i)
            batch_images = image_test[i:i+test_batch]
            batch_pred = sess.run(tf.argmax(output,1), feed_dict = {final_input: batch_images,is_training:False})
            for j in range(test_batch):
                predictions.append(batch_pred[j])
        test_classes = np.asarray([labels_unique[predictions[i]] for i in range(len(predictions))])
        test_df = pd.DataFrame(test_classes)
        test_df['id'] = pd.Series([i for i in range(1,len(predictions)+1)])
        test_df['label'] = test_df[test_df.columns[0]]
        test_df = test_df[['id','label']]
        test_df.to_csv('submission2.csv',index=False)
