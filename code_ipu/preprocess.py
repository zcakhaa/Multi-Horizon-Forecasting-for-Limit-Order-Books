import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import logging

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    all_label = []

    for i in range(lob.shape[1]):
        one_label = lob[:, i] - 1
        one_label = keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)

    return np.hstack(all_label)


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY

    
def create_dataset(encoder_input, decoder_input, encoder_target, batch_size, method, shuffle=False):
    train_pairs_ds = tf.data.Dataset.from_tensor_slices((encoder_input, decoder_input))
    train_pairs_ds = train_pairs_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    train_y_ds = tf.data.Dataset.from_tensor_slices(encoder_target)
    train_y_ds = train_y_ds.map(lambda d: (tf.cast(d, tf.float32)))

    train_ds = tf.data.Dataset.zip((train_pairs_ds, train_y_ds))

    if shuffle:
        train_ds = train_ds.shuffle(len(encoder_input))
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    #     train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    if method == 'train':
        return train_ds.repeat()

    if method == 'val':
        return train_ds

    if method == 'test':
        return train_ds

    if method == 'prediction':
        train_ds = tf.data.Dataset.from_tensor_slices((encoder_input, decoder_input))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
        return train_ds


def evaluation_metrics(real_y, pred_y):
    real_y = real_y[:len(pred_y)]
    logging.info('-------------------------------')

    for i in range(real_y.shape[1]):
        print(f'Prediction horizon = {i}')
        print(f'accuracy_score = {accuracy_score(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1))}')
        print(f'classification_report = {classification_report(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1), digits=4)}')
        print('-------------------------------')


def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    return decoder_input_data


# def get_label(data):
#     lob = data[-5:, :].T
#     return lob

# def create_dataset_single(x_train, y_train, batch_size, method):
#     train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True)
#     train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

#     if method == 'train':
#         return train_ds.repeat()

#     if method == 'val':
#         return train_ds

#     if method == 'test':
#         return train_ds

#     if method == 'prediction':
#         train_ds = tf.data.Dataset.from_tensor_slices((x_train))
#         train_ds = train_ds.batch(batch_size, drop_remainder=True)
#         train_ds = train_ds.map(lambda d: (tf.cast(d, tf.float32)))
#         return train_ds
