import numpy as np
import matplotlib.pyplot  as plt
import struct

class NumberClassifier(object):
    def __init__(self):
        self.train_data, self.train_labels = self.load_data(
            'data\\train-images.idx3-ubyte',
            'data\\train-labels.idx1-ubyte',
            type=True)
        self.test_data, self.test_labels = self.load_data(
            'data\\t10k-images.idx3-ubyte',
            'data\\t10k-labels.idx1-ubyte',
            type=True)
        self.w_i = np.random.uniform(-0.5, 0.5, (90, 784))
        self.b_i = np.zeros((90, 1))
        self.w_j = np.random.uniform(-0.5, 0.5, (10, 90))
        self.b_j = np.zeros((10, 1))
        #superparaments
        self.learn_rate = 0.1
        self.epochs = 5

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
            for in_layer, in_label in zip(self.train_data, self.train_labels):
                in_layer = in_layer.reshape(784, 1) / 255
                #正向传播
                hidden_layer = (np.dot(self.w_i, in_layer) + self.b_i) #中间层
                hidden_layer = 1 / (1 + np.exp(-hidden_layer)) #sigmoid
                out_layer = np.dot(self.w_j, hidden_layer) + self.b_j #输出层
                out_layer = 1 / (1 + np.exp(-out_layer)) #sigmoid

                #误差评估
                target = np.zeros((10, 1), dtype='uint8')
                target[in_label][0] = 1
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
            print(f"Accuracy of round {epoch + 1}: {round(correct_num / self.train_data.shape[0] * 100, 2)}%")
            self.corrct_num = 0 

    def recognize(self, img):
        img = img.reshape(784, 1) / 255
        hidden_layer = (np.dot(self.w_i, img) + self.b_i) #中间层
        hidden_layer = 1 / (1 + np.exp(-hidden_layer)) #sigmoid
        out_layer = np.dot(self.w_j, hidden_layer) + self.b_j #输出层
        out_layer = 1 / (1 + np.exp(-out_layer)) #sigmoid
        return np.argmax(out_layer)
    
    def validation(self):
        incorrect = []
        correct_num = 0
        for index, (img, label) in enumerate(zip(self.test_data, self.test_labels)):
            correct_num += int(self.recognize(img) == label)
            if(self.recognize(img) != label) :
                incorrect.append(index)
        print(f"Accuracy of Validation: {round(correct_num / self.test_data.shape[0] * 100, 2)}%")
        #错误识别样本展示20个
        p = np.random.choice(len(incorrect), size=min(20, len(incorrect)), replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(12, 12))  
        for i, idx in enumerate(p):  
            row = i // 5
            col = i % 5
            axes[row, col].imshow(self.test_data[incorrect[idx]].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Predicted: {self.recognize(self.test_data[incorrect[idx]])}\nActual: {self.test_labels[incorrect[idx]]}")
            axes[row, col].axis('off')
        plt.show()

    def presentation(self, index):
        plt.imshow(self.train_data[index].reshape(28, 28), cmap="Grays")
        plt.title(f"Predicted: {self.recognize(self.train_data[index])}")
            
    def predict(self):
        while True:
            index = int(input())
            if index < 0:
                break
            plt.imshow(self.train_data[index].reshape(28, 28), cmap="Grays")
            plt.title(f"Predicted: {self.recognize(self.train_data[index])}")
            plt.pause(0)

if __name__ == '__main__':
    classifier = NumberClassifier()
    classifier.train()
    classifier.validation()

    # p = np.random.choice(train_data.shape[0], size=20, replace=False)
    # samples_img = train_data[p]
    # samples_label = train_labels[p]

    # for label, img in zip(samples_label, samples_img) :
    #     print(label)
    #     plt.imshow(img, cmap="Grays")
    #     plt.pause(0.5)
    # plt.pause(0)
    
    
