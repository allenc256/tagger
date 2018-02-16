import tensorflow as tf


def _rnn_dropout(input_data, rate, training, recurrent=True, time_major=False):
    # sizes
    batch_size = tf.shape(input_data)[1 if time_major else 0]
    size = input_data.shape[-1].value

    # noise mask
    ns = None
    if recurrent:
        ns = [1, batch_size, size] if time_major else [batch_size, 1, size]

    # apply dropout
    return tf.layers.dropout(
        input_data, rate=rate, training=training, noise_shape=ns)


def _rnn_unidir(input_data, size,  dropout_rate,training, name='rnn_unidir',
                reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # sizes
        batch_size = tf.shape(input_data)[1]

        # GRU
        gru = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers=1, num_units=size, input_size=input_data.shape[-1].value)

        # variables
        # N.B., initializers below lead to better training
        gru_params = tf.get_variable(
            'gru_params',
            initializer=tf.random_uniform(
                [gru.params_size().eval()], -0.1, 0.1))
        gru_input_h = tf.get_variable(
            'gru_input_h', [1, 1, size], initializer=tf.zeros_initializer())

        # dropout
        d_in = _rnn_dropout(input_data, dropout_rate, training, time_major=True)

        # tile input states
        h_in = tf.tile(gru_input_h, [1, batch_size, 1])

        # compute GRU
        d_out, h_out = gru(d_in, h_in, gru_params)

        return d_out


def _rnn_bidir(input_data, input_lens, size, dropout_rate, training,
               name='rnn_bidir', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # reverse input
        d_in_fw = input_data
        d_in_bk = tf.reverse_sequence(
            input_data, input_lens, seq_axis=0, batch_axis=1)

        # RNN
        d_out_fw = _rnn_unidir(
            d_in_fw, size, dropout_rate, training, 'fw', reuse)
        d_out_bk = _rnn_unidir(
            d_in_bk, size, dropout_rate, training, 'bk', reuse)

        # reverse output
        d_out_bk = tf.reverse_sequence(
            d_out_bk, input_lens, seq_axis=0, batch_axis=1)

        # concat
        return tf.concat([d_out_fw, d_out_bk], axis=-1)


def rnn(input_data, input_lens, size, num_layers, dropout_rate, training,
        name='rnn', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # tranpose to time-major
        d = tf.transpose(input_data, perm=[1, 0, 2])
        d_out = []

        for i in range(num_layers):
            d = _rnn_bidir(
                d, input_lens, size, dropout_rate, training, 'layer_%d' % i,
                reuse)
            d_out.append(d)

        # concat outputs from all layers
        o = tf.concat(d_out, axis=-1)

        # transpose from time-major
        o = tf.transpose(o, perm=[1, 0, 2])

        return o
