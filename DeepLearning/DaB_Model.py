from keras.models import Model  # 從 Keras 中匯入 Model 類別，用來建立神經網路模型
from keras import layers, callbacks  # 匯入 Keras 的回調功能，用於訓練過程中的中斷或監控
from keras.layers import (  # 匯入 Keras 各種層的類別
    Input,  # 用於定義模型的輸入層
    Reshape,  # 用於重新定義張量的形狀
    Dense,  # 全連接層
    BatchNormalization,  # 批次正規化層，標準化輸入
    Activation,  # 激活函數層
    GlobalAveragePooling2D,  # 全局平均池化層，用於降維
    Conv2D,  # 卷積層
    LSTM,
    GlobalAveragePooling1D,
    Permute,
    add,  # 用於合併兩個張量
)
from keras.optimizers import Adam  # 匯入 Adam 優化器，用於調整學習率
from keras.regularizers import l2  # 匯入 l2 正規化，用於防止模型過擬合
import matplotlib.pyplot as plt  # 匯入 Matplotlib 用於畫圖
import numpy as np
import os  # 匯入 NumPy 和 os 模組，NumPy 用於數學運算，os 用於檔案操作
import keras


class DaB_BaseModel():
    def __init__(self, input_shape, args):
        self.args = args
        self.w, self.h, self.c = input_shape
        self.m = int((self.w+1)/2)
        self.n = int((self.h+1)/2)
        self.model_name = f""
        self.model_type = f""

    def predict(self, board):
        return self.model.predict(board)[0]

    def fit(self, data, batch_size, epochs):
        input_boards, target_policys = zip(*data)

        if (self.args['type'] == 0):
            input_boards = np.array([np.array(board).reshape(self.w, self.h)
                                    for board in input_boards])

        elif (self.args['type'] == 1):
            input_boards = np.array([np.array(board).reshape(self.w, self.h, self.c)
                                    for board in input_boards])

        elif (self.args['type'] == 2):
            input_boards = np.array([np.array(board).reshape(self.c, (self.w * self.h))
                                    for board in input_boards])

        target_policys = np.array(
            [np.array(policy).reshape(self.w * self.h) for policy in target_policys])

        print(f"Input boards shape: {input_boards.shape}")
        print(f"Target policies shape: {target_policys.shape}")

        history = self.model.fit(x=input_boards,
                                 y=[target_policys],
                                 batch_size=batch_size,
                                 epochs=epochs)

        return history

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_weights(self):
        self.model.save_weights(
            'models/'+self.model_type+'/' + self.model_name)
        print(f'Model saved to models/'+self.model_type+'/'+self.model_name)

    def load_weights(self):
        self.model.load_weights('models/'+self.model_type+'/'+self.model_name)

    def reset(self, confirm=False):
        if not confirm:
            raise Exception(
                'This operation would clear model weights. Pass confirm=True if really sure.')
        else:
            try:
                os.remove('models/'+self.model_type+'/'+self.model_name)
            except:
                pass
        print('Cleared')

    def plot_learning_curve(self, history):
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')

        file_path = f'training_log/{self.model_type}/{self.model_name.split(".h5")[0]}_loss_1.png'
        base, extension = os.path.splitext(file_path)
        base = base[:-2]
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{extension}"
            counter += 1

        plt.savefig(new_file_path)
        plt.close()


class DaB_CNN(DaB_BaseModel, keras.Model):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        super(DaB_BaseModel, self).__init__()

        self.model_name = f"CNN_model_{self.m}x{self.n}.h5"
        self.model_type = f"CNN"

        self.input_boards = Input(shape=(self.w, self.h, self.c))
        x_image = self.input_boards
        x_image = layers.Conv2D(filters=32, kernel_size=(5, 5),
                                activation="relu", padding="same")(x_image)
        x_image = layers.MaxPooling2D(
            pool_size=(2, 2), padding="same")(x_image)
        x_image = layers.Dropout(0.3)(x_image)

        x_image = layers.Conv2D(filters=32, kernel_size=(5, 5),
                                activation="relu", padding="same")(x_image)
        x_image = layers.MaxPooling2D(
            pool_size=(2, 2), padding="same")(x_image)
        x_image = layers.Dropout(0.3)(x_image)

        x_image = layers.Conv2D(filters=128, kernel_size=(3, 3),
                                activation="relu", padding="same")(x_image)

        gap1 = GlobalAveragePooling2D()(x_image)
        self.pi = Dense(self.w*self.h,
                        activation='softmax', name='pi')(gap1)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(
            loss=['categorical_crossentropy'], optimizer=Adam(0.002))


class DaB_ResNet(DaB_BaseModel):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)

        self.model_name = f"Resnet_model_{self.m}x{self.n}.h5"
        self.model_type = f"Resnet"

        self.input_boards = Input(shape=(self.w, self.h))
        x_image = Reshape((self.w, self.h) + (1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs=x_image,
                                    num_res_blocks=2)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        self.pi = Dense(self.w*self.h,
                        activation='softmax', name='pi')(gap1)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(
            loss=['categorical_crossentropy'], optimizer=Adam(0.002))

    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for i in range(num_res_blocks):
            resnet = self.resnet_layer(inputs=x,
                                       num_filter=128)
            resnet = self.resnet_layer(inputs=resnet,
                                       num_filter=128,
                                       activation=None)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        return x

    def resnet_layer(self, inputs, num_filter=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):

        conv = Conv2D(num_filter,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)

        else:
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)

        return x


class DaB_LSTM(DaB_BaseModel):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)

        self.model_name = f"LSTM_model_{self.m}x{self.n}.h5"
        self.model_type = f"LSTM"

        self.input_boards = Input(shape=(self.c, (self.w*self.h)))
        x = self.input_boards
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = GlobalAveragePooling1D()(x)
        self.pi = Dense(input_shape[0] * input_shape[1],
                        activation='softmax', name='pi')(x)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(
            loss=['categorical_crossentropy'], optimizer=Adam(0.002))
