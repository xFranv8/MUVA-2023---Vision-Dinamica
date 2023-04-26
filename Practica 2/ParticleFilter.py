from BackSubstraction import BackSubstraction
import cv2
import natsort
import numpy as np
import os
from Particle import Particle
import random

PATH = "Foreground/"


def extract_foreground():
    substractor = BackSubstraction(PATH)
    images = sorted(os.listdir(PATH))

    if not os.path.exists("Foreground"):
        os.makedirs("Foreground")

    for i in range(len(images)):
        fg = substractor.substract(images[i])
        cv2.imwrite(f"Foreground/{images[i]}", fg)


def generate_particles(N: int) -> list:
    particles: list = []
    for i in range(N):
        x: int = random.randint(0, 300)
        y: int = random.randint(0, 220)
        w: float = 1 / N

        particles.append(Particle(x, y, w))

    return particles


def calc_particle_area(image: np.ndarray, particle: Particle) -> int:
    x: int = particle.x
    y: int = particle.y
    area: int = image[y: y + 31, x: x + 31].sum()

    return area


def get_total_count(image: np.ndarray, particles: list) -> int:
    total_area: int = 0
    for p in particles:
        area = calc_particle_area(image, p)
        total_area += area

    return total_area


def get_max_window(random_positions: list, image: np.array) -> tuple:
    max_sum: int = 0
    x: int = 0
    y: int = 0
    for j, i in random_positions:
        particle: int = image[i: i + 31, j: j + 31].sum()
        if particle > max_sum:
            max_sum = particle
            x = j
            y = i

    return x, y


def main():
    """image: np.array = cv2.imread("Foreground/10.jpg")
    real_image: np.array = cv2.imread("Secuencia/10.jpg")"""
    particles: list[Particle] = generate_particles(60)
    # x, y = get_max_window(particles, image)

    kernel = np.ones((9, 9), np.float32)

    images: list = natsort.natsorted(os.listdir(PATH))
    for img in images:
        print(img)
        image: np.ndarray = cv2.imread(PATH + '/' + img).astype(np.float32)

        image = cv2.erode(image, kernel, 9)
        image = cv2.dilate(image, kernel, 9)

        total_area: int = get_total_count(image, particles)

        if total_area == 0:
            for p in particles:
                cv2.rectangle(image, (p.x, p.y), (p.x + 31, p.y + 31),
                              (0, 255, 0), 1)

            cv2.imshow("", np.uint8(image))
            if cv2.waitKey(0) == ord('c'):
                particles = generate_particles(80)
                continue

        # Update weigths and create accumulated vector of weights
        aux: float = 0.0
        accumulated_weights: list = []
        for p in particles:
            area = calc_particle_area(image, p)

            p.weight = area / (total_area + 1e-6)

            aux += p.weight
            accumulated_weights.append(aux)

        """i: int = 0
        random_number = np.random.uniform(0, 1)
        while i < len(accumulated_weights) and accumulated_weights[i] <= random_number:
            random_number = random.uniform(0, 1)
            i += 1"""

        new_particles: list[Particle] = []
        for i in range(len(particles)):
            p: Particle = None
            rand_number: float = np.random.uniform(0, 1)
            for j in range(len(accumulated_weights)):
                if accumulated_weights[j] > rand_number:
                    p: Particle = particles[j]
                    break

            if p is not None:
                # Dispersion
                dispersionx: int = int(p.x + 25 * np.random.normal(0, 1) + p.vx)
                dispersiony: int = int(p.y + 25 * np.random.normal(0, 1) + p.vy)

                p.vx += np.random.normal(0, 1)
                p.vy += np.random.normal(0, 1)

                if dispersionx < 0:
                    dispersionx = 0
                elif dispersionx + 31 > 320:
                    dispersionx = 320 - 31

                if dispersiony < 0:
                    dispersiony = 0
                elif dispersiony + 31 > 240:
                    dispersiony = 240 - 31

                particle: Particle = Particle(dispersionx, dispersiony, p.weight)

                particle.vx = p.vx
                particle.vy = p.vy

                new_particles.append(particle)

        max_window: tuple = get_max_window([(p.x, p.y) for p in new_particles], image)

        for p in new_particles:
            cv2.rectangle(image, (p.x, p.y), (p.x + 31, p.y + 31),
                          (0, 255, 0), 1)

        cv2.rectangle(image, (max_window[0], max_window[1]), (max_window[0] + 31, max_window[1] + 31), (0, 0, 255), 1)

        cv2.imshow("", np.uint8(image))
        if cv2.waitKey(0) == ord('c'):
            continue

    """cv2.rectangle(real_image, (x, y), (x+31, y+31), (0, 0, 255), 2)
    cv2.imshow("", real_image)
    cv2.waitKey(0)"""


if __name__ == '__main__':
    main()
