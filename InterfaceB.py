import tensorflow as tf
import tensorflow_datasets as tfds

# https://www.tensorflow.org/datasets/keras_example
class InterfaceB(object):
    def __init__(self, batch_size: int) -> None:
        """Loading script"""
        (self.ds_train, self.ds_test), self.ds_info = tfds.load(
                                                    'mnist',
                                                    split=['train', 'test'],
                                                    shuffle_files=True,
                                                    as_supervised=True,
                                                    with_info=True,
                                                )
        self.batch_size = batch_size
    def preprocess(self):
        def normalize_and_flatten_img(image, label):
            """Normalizes images: `uint8` -> `float32`."""
            return tf.cast(image, tf.float32) / 255., label

        ds_train = self.ds_train.map(
            normalize_and_flatten_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(self.ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(self.batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = self.ds_test.map(
        normalize_and_flatten_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        return (ds_train, ds_test)


def test():
    batch_size = 10
    dataset = InterfaceB(batch_size)
    train_set, test_set = dataset.preprocess()
    print(test_set)

if __name__ == "__main__": test()


