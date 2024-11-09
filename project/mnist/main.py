import numpy as np
import matplotlib.pyplot  as plt
import struct

class NumberClassifier(object):
    def __init__(self):
        self.train_data, self.train_lables = self.load_data(
            'data\\train-images.idx3-ubyte',
            'data\\train-labels.idx1-ubyte',
            type=True)
        self.test_data, self.test_lables = self.load_data(
            'data\\t10k-images.idx3-ubyte',
            'data\\t10k-labels.idx1-ubyte',
            type=True)
        self.w_i = np.random.uniform(-0.5, 0.5, (90, 784))
        self.b_i = np.zeros((90, 1))
        self.w_j = np.random.uniform(-0.5, 0.5, (10, 90))
        self.b_j = np.zeros((10, 1))
        #superparaments
        self.learn_rate = 0.1
        self.epochs = 10

    def load_data(self, images_path, label_path, type=False):
        with open(images_path, 'rb') as f:  
            _, image_num, rows, cols = struct.unpack('>IIII', f.read(16))
            if type:
                train_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(image_num, -1) #转为一维
            else:
                train_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, rows, cols) #正常图像
        with open(label_path, 'rb') as f:  
            _, label_num = struct.unpack('>II', f.read(8))  
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)
        return train_data, train_labels

    def train(self):
        for epoch in range(self.epochs):
            correct_num = 0
            for in_layer, in_lable in zip(self.train_data, self.train_lables):
                in_layer = in_layer.reshape(784, 1) / 255
                #正向传播
                hidden_layer = (np.dot(self.w_i, in_layer) + self.b_i) #中间层
                hidden_layer = 1 / (1 + np.exp(-hidden_layer)) #sigmoid
                out_layer = np.dot(self.w_j, hidden_layer) + self.b_j #输出层
                out_layer = 1 / (1 + np.exp(-out_layer)) #sigmoid

                #误差评估
                target = np.zeros((10, 1), dtype='uint8')
                target[in_lable][0] = 1
                loss = np.sum((out_layer - target) ** 2) / 10 #均方误差函数
                correct_num += int(np.argmax(target) == np.argmax(out_layer))

                #反向传播
                delta_out_layer = (out_layer - target) / 20 
                delta_out_layer = delta_out_layer * out_layer * (1 - out_layer) #sigmoid求导

                #输出层->隐藏层
                delta_hidden_layer = np.dot(self.w_j.T, delta_out_layer)
                self.w_j -= self.learn_rate * np.dot(delta_out_layer, hidden_layer.T) 
                self.b_j -= self.learn_rate * delta_out_layer

                #隐藏层->输入层
                delta_hidden_layer = delta_hidden_layer * hidden_layer * (1 - hidden_layer) #sigmoid求导
                self.w_i -= self.learn_rate * np.dot(delta_hidden_layer, in_layer.T)
                self.b_i -= self.learn_rate * delta_hidden_layer
            print(f"Accurancy of round {epoch + 1}: {round(correct_num / self.train_data.shape[0] * 100, 2)}%")
            self.corrct_num = 0 

    def recongnize(self, img):
        img = img.reshape(784, 1) / 255
        hidden_layer = (np.dot(self.w_i, img) + self.b_i) #中间层
        hidden_layer = 1 / (1 + np.exp(-hidden_layer)) #sigmoid
        out_layer = np.dot(self.w_j, hidden_layer) + self.b_j #输出层
        out_layer = 1 / (1 + np.exp(-out_layer)) #sigmoid
        return np.argmax(out_layer)
    
    def validation(self):
        correct_num = 0
        for img, lable in zip(self.test_data, self.test_lables):
            correct_num += int(self.recongnize(img) == lable)
            # if(self.recongnize(img) != lable) :
            #     plt.imshow(img.reshape(28, 28), cmap="Grays")
            #     plt.title(f"find {self.recongnize(img)} except {lable}")
            #     plt.pause(2)
        print(f"Accurancy of Validation: {round(correct_num / self.test_data.shape[0] * 100, 2)}%")

    def presentation(self, index):
        plt.imshow(self.train_data[index].reshape(28, 28), cmap="Grays")
        plt.title(f"It's may be {self.recongnize(self.train_data[index])}")
            
    def predict(self):
        while True:
            index = int(input())
            if index < 0:
                break
            plt.imshow(self.train_data[index].reshape(28, 28), cmap="Grays")
            plt.title(f"It's may be {self.recongnize(self.train_data[index])}")
            plt.pause(0)
        pass
    



if __name__ == '__main__':
    classifier = NumberClassifier()
    classifier.train()
    classifier.validation()
    # print(classifier.train_data.shape)
    # np.reshape(temp, (-1, 28, 28))
    # print(temp.shape)

    # p = np.random.choice(train_data.shape[0], size=20, replace=False)
    # samples_img = train_data[p]
    # samples_label = train_lables[p]

    # for label, img in zip(samples_label, samples_img) :
    #     print(label)
    #     plt.imshow(img, cmap="Grays")
    #     plt.pause(0.5)
    # plt.pause(0)
    
    
