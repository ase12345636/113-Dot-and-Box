from keras.models import Model  # 從 Keras 中匯入 Model 類別，用來建立神經網路模型
from keras import callbacks  # 匯入 Keras 的回調功能，用於訓練過程中的中斷或監控
from keras.layers import (  # 匯入 Keras 各種層的類別
    Input,  # 用於定義模型的輸入層
    Reshape,  # 用於重新定義張量的形狀
    Dense,  # 全連接層
    BatchNormalization,  # 批次正規化層，標準化輸入
    Activation,  # 激活函數層
    GlobalAveragePooling2D,  # 全局平均池化層，用於降維
    Conv2D,  # 卷積層
    add,  # 用於合併兩個張量
)
from keras.optimizers import Adam  # 匯入 Adam 優化器，用於調整學習率
from keras.regularizers import l2  # 匯入 l2 正規化，用於防止模型過擬合
import matplotlib.pyplot as plt  # 匯入 Matplotlib 用於畫圖
import numpy as np, os  # 匯入 NumPy 和 os 模組，NumPy 用於數學運算，os 用於檔案操作

class DaB_ResNet():  # 定義一個 DaB_ResNet 類別
    def __init__(self, input_shape):  # 初始化方法，輸入棋盤形狀
        print(input_shape)  # 打印輸入形狀
        self.input_shape = input_shape  # 將 input_shape 存到物件屬性中
        m, n = input_shape  # 分別取得行和列
        m = int((m+1)/2)  # 計算 m 為棋盤高度的線數
        n = int((n+1)/2)  # 計算 n 為棋盤寬度的線數
        self.model_name = f"model_{m}x{n}.h5"  # 設定模型名稱
        self.input_boards = Input(shape=input_shape)  # 定義輸入層，形狀為棋盤大小
        x_image = Reshape(input_shape + (1,))(self.input_boards)  # 將輸入的形狀重塑，增加一個維度以符合卷積網路要求
        resnet_v12 = self.resnet_v1(inputs=x_image, num_res_blocks=2)  # 建立 ResNet，使用 2 個殘差區塊
        gap1 = GlobalAveragePooling2D()(resnet_v12)  # 使用全局平均池化層
        self.pi = Dense(input_shape[0] * input_shape[1], activation='softmax', name='pi')(gap1)  # 定義輸出層，輸出為棋盤的每個動作概率
        self.model = Model(inputs=self.input_boards, outputs=[self.pi])  # 定義模型，輸入為棋盤，輸出為動作概率
        self.model.compile(loss=['categorical_crossentropy'], optimizer=Adam(0.002))  # 編譯模型，使用 Adam 優化器和交叉熵損失函數

    def predict(self, board):  # 定義一個預測方法，輸入棋盤狀態，輸出預測動作
        return self.model.predict(board)[0]

    def fit(self, data, batch_size, epochs):  # 定義模型訓練方法，輸入數據，批次大小和訓練次數
        m,n = self.input_shape  # 取得棋盤的行和列
        
        input_boards, target_policys = zip(*data)  # 解壓數據，分為輸入的棋盤和目標策略
        
        input_boards = np.array([np.array(board).reshape(m, n) for board in input_boards])  # 將棋盤數據轉換為 numpy 陣列並重塑為合適的形狀
        
        target_policys = np.array([np.array(policy).reshape(m * n) for policy in target_policys])  # 將目標策略展開為一維
        
        print(f"Input boards shape: {input_boards.shape}")  # 打印棋盤的形狀
        print(f"Target policies shape: {target_policys.shape}")  # 打印策略的形狀
        
        history = self.model.fit(x=input_boards, y=[target_policys], batch_size=batch_size, epochs=epochs)  # 訓練模型
        return history  # 返回訓練過程歷史記錄
        
    def set_weights(self, weights):  # 定義一個設定權重的方法
        self.model.set_weights(weights)
    
    def get_weights(self):  # 定義一個獲取權重的方法
        return self.model.get_weights()
    
    def save_weights(self):  # 定義一個保存權重的方法
        self.model.save_weights('models/' + self.model_name)  # 將權重存檔
        print(f'Model saved to [models/{self.model_name}]')  # 打印保存成功訊息
    
    def load_weights(self):  # 定義一個讀取權重的方法
        self.model.load_weights('models/'+self.model_name)
    
    def reset(self, confirm=False):  # 定義一個重置方法，需提供確認參數以避免意外清除
        if not confirm:
            raise Exception('This operation would clear model weights. Pass confirm=True if really sure.')  # 若未確認，則拋出異常
        else:
            try:
                os.remove('models/'+self.model_name)  # 嘗試刪除模型權重檔案
            except:
                pass
        print('Cleared')  # 打印清除訊息
         
    def resnet_v1(self, inputs, num_res_blocks):  # 定義 ResNet 模型的方法
        x = inputs  # 輸入數據
        for i in range(num_res_blocks):  # 迭代指定次數的殘差區塊
            resnet = self.resnet_layer(inputs=x, num_filter=128)  # 第一次卷積層，過濾器數量為128
            resnet = self.resnet_layer(inputs=resnet, num_filter=128, activation=None)  # 第二次卷積層，沒有激活函數
            resnet = add([resnet, x])  # 將輸入與卷積層的輸出相加，形成殘差結構
            resnet = Activation('relu')(resnet)  # 通過 ReLU 激活函數
            x = resnet  # 更新輸出為下一次迭代的輸入
        return x  # 返回最後的輸出

    def resnet_layer(self, inputs, num_filter=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):  
        # 定義殘差網路層，包含卷積、批次正規化和激活函數
        conv = Conv2D(num_filter, 
                    kernel_size=kernel_size, 
                    strides=strides, 
                    padding='same',
                    use_bias=False,  
                    kernel_regularizer=l2(1e-4))  # 定義卷積層，使用 L2 正規化
        x = inputs  # 輸入數據
        if conv_first:  # 如果卷積優先執行
            x = conv(x)  # 卷積操作
            if batch_normalization:  # 如果批次正規化啟用
                x = BatchNormalization(axis=3)(x)  # 進行批次正規化
            if activation is not None:  # 如果激活函數啟用
                x = Activation(activation)(x)  # 進行激活
        else:  # 如果激活優先執行
            if batch_normalization:
                x = BatchNormalization(axis=3)(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)  # 卷積操作
        return x  # 返回卷積層的輸出

    def plot_learning_curve(self, history):
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        
        file_path = f'training_log/ResNet_{self.model_name.split(".h5")[0]}_loss_1.png'
        base, extension = os.path.splitext(file_path)
        base = base[:-2]
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = f"{base}_{counter}{extension}"
            counter += 1
        plt.savefig(new_file_path)
        #plt.show()
        plt.close()

        #print(f"Learning curve saved to {new_file_path}")