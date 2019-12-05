import tensorflow_io.ffmpeg as ffmpeg
import tensorflow as tf
from time import time
# tf.estimator.export.ServingInputReceiver
# tf.enable_eager_execution()
def main():
    # 368,
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():


    # filename = '/Users/nicholastancredi/downloaded_videos/original_video-output.mp4'
    filename = '/root/openpose-plus/downloaded_videos/original_video.mp4'

    dataset = ffmpeg.VideoDataset(filename, batch=1).map(
        map_func=tf.keras.layers.Lambda(lambda image: tf.image.convert_image_dtype(
            tf.image.resize_with_pad(
                image, 656, 368
            ),
            dtype=tf.float32
        )),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    # num_parallel_calls
    try:
        resize_with_pad = tf.image.resize_with_pad
    except:
        resize_with_pad = tf.image.resize_image_with_pad

    # dataset = ffmpeg.VideoDataset(filename, batch=8).map(map_func=tf.keras.layers.Lambda(parse_fn))

    # with tf.device('/gpu:0'):
    #     inputs = tf.keras.layers.Input(shape=(
    #         656,
    #         368,
    #         3
    #     ), name="Frames")
    #
    #     outputs = inputs
    #     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    shape = (
        # 656,
        # 368,
        None,
        None,
        3
    )
    inputs = tf.keras.layers.Input(shape=shape, name="Frames")
    outputs = inputs
    # outputs = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    # outputs = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.MaxPool2D()(outputs)
    # outputs = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.MaxPool2D()(outputs)
    # outputs = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(outputs)
    # outputs = tf.keras.layers.MaxPool2D()(outputs)
    # outputs = tf.keras.layers.Conv2D(512, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(512, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(512, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(512, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(256*4, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(256*4, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(256*4, 3)(outputs)
    # outputs = tf.keras.layers.Conv2D(256*4, 3)(outputs)
    s = time()
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(time() - s)
    # vgg19 = tf.keras.applications.vgg19

    def parse_fn(image):
        # start = time()
        # print('convert_image_dtype', time() - start)
        return tf.image.convert_image_dtype(
            tf.image.resize_with_pad(
                image, 656, 368
            ),
            dtype=tf.float32
        )

    # dataset = ffmpeg.VideoDataset(filename, batch=1).range(0, 1)
    dataset = ffmpeg.VideoDataset(filename, batch=1).map(
        map_func=tf.keras.layers.Lambda(lambda image: tf.image.convert_image_dtype(
            tf.image.resize_with_pad(
                image, 656, 368
            ),
            dtype=tf.float32
        )),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)

    # model = tf.keras.estimator.model_to_estimator(model)

    s = time()
    # process dataset elements
    pred = model.predict(dataset)
    print(pred.shape)
    print(time() - s)
    # import numpy as np
    #
    # # Generate dummy data
    # x_train = np.random.random((100, 100, 100, 3))
    # y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    # x_test = np.random.random((20, 100, 100, 3))
    # y_test = tf.keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
    #
    # model = tf.keras.models.Sequential()
    # # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # # this applies 32 convolution filters of size 3x3 each.
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.25))
    #
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(tf.keras.layers.Dropout(0.25))
    #
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(10, activation='softmax'))
    #
    # sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #
    # model.summary()
    # model.fit(x_train, y_train, batch_size=32, epochs=2)
    # score = model.evaluate(x_test, y_test, batch_size=32)
    # print('score', score)
    # prediction = model.predict(dataset)
    # print(prediction)
    # print(len(prediction))
    # print(prediction.shape)
    # # model.compile(loss='mse', optimizer='sgd')
    # # model.fit(dataset, epochs=2)
    # # prediction = vgg19.predict(dataset)
    # # print("prediction", prediction)


if __name__ == '__main__':
    main()