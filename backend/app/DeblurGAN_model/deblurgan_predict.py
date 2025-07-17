import os
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()


class DeblurGANPredictor:
    def __init__(self, checkpoint_dir):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_dir}")
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(self.sess, ckpt)
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
