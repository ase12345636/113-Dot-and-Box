from keras import layers
import keras


class BaseModel(keras.Model):
    def __init__(self, w, h, c):
        super(BaseModel, self).__init__()

        self.w = w
        self.h = h
        self.c = c


class CNN(BaseModel):
    def __init__(self, w, h, c):
        super(CNN, self).__init__(w, h, c)

        self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7),
                                   activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(filters=64, kernel_size=(5, 5),
                                   activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")
        self.dropout2 = layers.Dropout(0.3)

        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3, 3),
                                   activation="relu", padding="same")
        self.pool3 = layers.MaxPooling2D(
            pool_size=(2, 2), padding="same")
        self.dropout3 = layers.Dropout(0.3)

        self.gap = layers.GlobalAveragePooling2D()
        self.pi = layers.Dense(w*h, activation='softmax', name='pi')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.gap(x)
        predict = self.pi(x)

        return predict

    def build_graph(self):
        x = layers.Input(shape=(self.w, self.h, self.c))
        return keras.Model(inputs=[x], outputs=self.call(x))


class ResNet(BaseModel):
    def __init__(self, w, h, c):
        super(ResNet, self).__init__(w, h, c)

        self.num_res_block = 3

        self.reshape = layers.Reshape((self.w, self.h) + (1,))

        self.conv = [layers.Conv2D(filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(1e-4))
                     for i in range(2*self.num_res_block)]

        self.batch_norm = [layers.BatchNormalization(axis=3)
                           for i in range(2*self.num_res_block)]

        self.actfun = [layers.Activation('relu')
                       for i in range(2*self.num_res_block)]

        self.gap = layers.GlobalAveragePooling2D()
        self.pi = layers.Dense(w*h, activation='softmax', name='pi')

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x, activation=True, cnt=i*2)
            resnet = self.resnet_layer(inputs=resnet, cnt=i*2+1)
            resnet = layers.add([resnet, x])
            resnet = self.actfun[i*2+1](resnet)
            x = resnet

        return x

    def resnet_layer(self, inputs, activation=False, batch_normalization=True, conv_first=True, cnt=0):

        x = inputs
        if conv_first:
            x = self.conv[cnt](x)
            if batch_normalization:
                x = self.batch_norm[cnt](x)
            if activation:
                x = self.actfun[cnt](x)

        else:
            if batch_normalization:
                x = self.batch_norm[cnt](x)
            if activation:
                x = self.actfun[cnt](x)
            x = self.conv[cnt](x)

        return x

    def call(self, x):
        x = self.reshape(x)
        x = self.resnet_v1(inputs=x,
                           num_res_blocks=self.num_res_block)
        x = self.gap(x)
        predict = self.pi(x)

        return predict

    def build_graph(self):
        x = layers.Input(shape=(self.w, self.h))
        return keras.Model(inputs=[x], outputs=self.call(x))


class LSTM(BaseModel):
    def __init__(self, w, h, c):
        super(LSTM, self).__init__(w, h, c)

        self.LSTM1 = layers.LSTM(128, return_sequences=True)
        self.LSTM2 = layers.LSTM(64, return_sequences=True)
        self.gap = layers.GlobalAveragePooling1D()
        self.pi = layers.Dense(w*h,
                               activation='softmax', name='pi')

    def call(self, x):
        x = self.LSTM1(x)
        x = self.LSTM2(x)

        x = self.gap(x)
        predict = self.pi(x)

        return predict

    def build_graph(self):
        x = layers.Input(shape=(self.c, (self.w*self.h)))
        return keras.Model(inputs=[x], outputs=self.call(x))
