from BackSubstraction import BackSubstraction
import cv2
import numpy as np
import os
import random


def extract_foreground():
    path = "Secuencia/"
    substractor = BackSubstraction(path)
    images = sorted(os.listdir(path))

    if not os.path.exists("Foreground"):
        os.makedirs("Foreground")

    for i in range(len(images)):
        fg = substractor.substract(images[i])
        cv2.imwrite(f"Foreground/{images[i]}", fg)


def generate_windows(N: int) -> list:
    random_positions: list = []
    for i in range(N):
        x: int = random.randint(0, 300)
        y: int = random.randint(0, 220)

        random_positions.append((x, y))

    return random_positions


def get_max_window(random_positions: list, image: np.array) -> tuple:
    max_sum: int = 0
    x: int = 0
    y: int = 0
    for i, j in random_positions:
        particle: int = image[i: i+31, j: j+31].sum()
        if particle > max_sum:
            max_sum = particle
            x = j
            y = i

    return x, y


def main():
    image: np.array = cv2.imread("Foreground/10.jpg")
    real_image: np.array = cv2.imread("Secuencia/10.jpg")
    particles: list = generate_windows(50)
    x, y = get_max_window(particles, image)

    cv2.rectangle(real_image, (x, y), (x+31, y+31), (0, 0, 255), 2)
    cv2.imshow("", real_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
