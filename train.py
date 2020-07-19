import random
import cv2
import numpy as np
from main import n

epochs = 20000

def train(epochs):
    for i in range(epochs):
        number = random.randint(0, 17)
        if number == 1 or number == 2 or number == 3:
            rightAnswers = np.array([[1, 0]])
        elif number == 4 or number == 5 or number == 6:
            rightAnswers = np.array([[0, 1]])
        elif 6 < number < 15:
            rightAnswers = np.array([[1, 1]])
        elif number == 0 or number == 16 or number == 17:
            rightAnswers = np.array([[0, 0]])

        image = cv2.imread("dataset/test/" + str(number) + ".png", 0).flatten() / 255

        n.train(image, rightAnswers)
