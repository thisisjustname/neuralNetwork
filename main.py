from Network import NeuralNetwork
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

input_nodes = 9
hidden_nodes = 5
output_nodes = 2
learning_rate = 0.2


n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

ppi = np.array([[0.3, 0.3]])

epochs = 1111

for i in range(epochs):
    a = np.random.permutation(18)
    for pro in a:
        number = pro
        # number = random.randint(0, 17)
        if number == 1 or number == 2 or number == 3:
            rightAnswers = np.array([[1, 0]])
        elif number == 4 or number == 5 or number == 6:
            rightAnswers = np.array([[0, 1]])
        elif 6 < number < 16 or number == 17:
            rightAnswers = np.array([[1, 1]])
        elif number == 0 or number == 16:
            rightAnswers = np.array([[0, 0]])

            # print('rightAnswers = ', rightAnswers)

        image = cv2.imread("dataset/test/" + str(number) + ".png", 0).flatten() / 255
        if pro == 15 and i % 25 == 0:
            ppi = np.vstack((ppi, n.train(image, rightAnswers).T))
            plt.clf()
            plt.xlabel(i * 17)
            plt.plot(ppi)
            plt.draw()  # Должно быть это, а в итоге не это, приходится каждый раз открывать картинку
            plt.pause(0.0000000000000000000001)
        else:
            n.train(image, rightAnswers)

plt.title('lr = ' + str(learning_rate))
plt.ioff()
plt.show()

for i in range(18):
    image = cv2.imread("dataset/test/" + str(i) + ".png", 0).flatten() / 255
    print(n.query(image))

np.save('./weights/example_1', n.synaptic_weight1)
np.save('./weights/example_2', n.synaptic_weight2)