import tensorflow as tf
import numpy as np
from PIL import Image


class DeblurGANPredictor:
    def __init__(self, checkpoint_dir):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(
                tf.train.latest_checkpoint(checkpoint_dir) + '.meta')
            saver.restore(
                self.sess, tf.train.latest_checkpoint(checkpoint_dir))
            self.input_image = self.graph.get_tensor_by_name('input_image:0')
            self.output_image = self.graph.get_tensor_by_name(
                'generator/output_image:0')

    def predict(self, pil_image):
        img = np.array(pil_image.resize((256, 256))).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        with self.graph.as_default():
            output = self.sess.run(self.output_image, feed_dict={
                                   self.input_image: img})
        output_img = (output[0] * 255).astype(np.uint8)
        return Image.fromarray(output_img)
