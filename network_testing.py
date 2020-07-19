from Network import NeuralNetwork
import cv2
import numpy as np
import matplotlib.pyplot as plt

n = NeuralNetwork(9, 5, 2, 0.3)

n.synaptic_weight1 = np.load('./weights/example_1.npy')
n.synaptic_weight2 = np.load('./weights/example_2.npy')


image = cv2.imread("dataset/test/test2.0.png", 0)

inputs = image.flatten()
inputs = np.asfarray(inputs / 255.0 * 0.99) + 0.01
answers = n.query(inputs)
print(str(answers[0]) + str(answers[1]))

plt.imshow(image)
plt.xlabel('test2.0.png')
plt.title(str(answers[0]) + '  --  ' + str(answers[1]))
plt.show()


for i in range(18):
    image = cv2.imread("dataset/test/" + str(i) + ".png", 0)

    inputs = image.flatten()
    inputs = np.asfarray(inputs / 255.0 * 0.99) + 0.01
    answers = n.query(inputs)
    print(str(answers[0]) + str(answers[1]))

    plt.imshow(image)
    plt.xlabel(str(i) + '.png')
    plt.title(str(answers[0]) + '  --  ' + str(answers[1]))
    plt.show()
