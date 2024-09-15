from tensorflow.keras.models import Model
from keras import callbacks
from tensorflow.keras.layers import (
    Input,
    Reshape,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    Conv2D,
    add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np, os

class DaB_ResNet():
    def __init__(self, input_shape=(5, 5)):
        self.model_name = 'model_' + ('x'.join(list(map(str, input_shape)))) + '.h5'
        self.input_boards = Input(shape=input_shape)
        x_image = Reshape(input_shape + (1,))(self.input_boards)
        resnet_v12 = self.resnet_v1(inputs=x_image, num_res_blocks=2)
        gap1 = GlobalAveragePooling2D()(resnet_v12)
        self.pi = Dense(input_shape[0] * input_shape[1], activation='softmax', name='pi')(gap1)
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])
        self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.002))
        
    def predict(self, board):
        board = np.expand_dims(board, axis=0).astype('float32')
        return self.model.predict(board)[0]
    
    def fit(self, data, batch_size, epochs):
        input_boards, target_policys = zip(*data)
        input_boards = np.array(input_boards)
        target_policys = np.array(target_policys)
        history = self.model.fit(x=input_boards, y=[target_policys], batch_size=batch_size, epochs=epochs)
        return history
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def save_weights(self):
        self.model.save_weights('models/' + self.model_name)
        print(f'Model saved to [models/{self.model_name}]')
    
    def load_other_weights(self, n):
        self.model.load_weights('models/ResNet_model_10x10_' + str(n) + '.h5')
        
    def load_weights(self):
        self.model.load_weights('models/'+self.model_name)
    
    def reset(self, confirm=False):
        if not confirm:
            raise Exception('this operate would clear model weight, pass confirm=True if really sure')
        else:
            try:
                os.remove('models/'+self.model_name)
            except:
                pass
        print('cleared')
         
    def resnet_v1(self, inputs, num_res_blocks):
        x = inputs
        for  i in range(1):
            resnet = self.resnet_layer(inputs = x, num_filter = 128)    
            resnet = self.resnet_layer(inputs = resnet, num_filter = 128, activation = None)     
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if(i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 256,strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 256)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 256, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 256, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet

        for i in range(2):
            if(i == 0):
                resnet = self.resnet_layer(inputs = x, num_filter = 512,strides=2)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
            else:
                resnet = self.resnet_layer(inputs = x, num_filter = 512)
                resnet = self.resnet_layer(inputs = resnet, num_filter = 512, activation = None)
            if(i == 0):
                x = self.resnet_layer(inputs = x, num_filter = 512, strides=2)
            resnet = add([resnet, x])
            resnet = Activation('relu')(resnet)
            x = resnet
        return x

    def resnet_layer(self, inputs, num_filter = 16, kernel_size = 3, strides = 1, activation = 'relu', batch_normalization = True, conv_first = True, padding = 'same'):
        
        conv = Conv2D(num_filter, 
                    kernel_size = kernel_size, 
                    strides = strides, 
                    padding = padding,
                    use_bias = False,  
                    kernel_regularizer = l2(1e-4))
        
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

    def plot_learning_curve(self,history):
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        
        # 检查文件是否存在，并生成唯一文件名
        file_path = '/ResNet_log/ResNet_loss_1.png'
        base, extension = os.path.splitext(file_path)
        base = base[0:len(base)-2]
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{extension}"
            counter += 1
        # 保存图像到文件
        plt.savefig(new_file_path)
        plt.close()  # 关闭图像，释放内存

        print(f"Learning curve saved to {new_file_path}")