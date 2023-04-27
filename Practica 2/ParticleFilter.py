from BackSubstraction import BackSubstraction
import cv2
import natsort
import numpy as np
import os
from Particle import Particle
import random

PATH = "Secuencia/"


class ParticleFilter:
    def __init__(self, PATH: str):
        self.__particles: list = []
        self.__generate_particles()

        self.__substractor: BackSubstraction = BackSubstraction(PATH)
        self.__kernel = np.ones((9, 9), np.float32)

        cv2.namedWindow("All particles")
        cv2.moveWindow("All particles", 0, 0)

        cv2.namedWindow("Foreground")
        cv2.moveWindow("Foreground", 320, 0)

        cv2.namedWindow("Result")
        cv2.moveWindow("Result", 320*2, 0)

    def __generate_particles(self) -> None:
        """
        This method generates N particles with random positions and same weight for each particle

        @param N:
        @return None:
        """
        N: int = 40
        particles: list = []
        for i in range(N):
            x: int = random.randint(0, 280)
            y: int = random.randint(0, 200)
            w: float = 1 / N

            particles.append(Particle(x, y, w))

        self.__particles = particles

    def __calc_particle_area(self, image: np.ndarray, particle: Particle) -> int:
        """
        This method calculates the area of a particle

        @param image:
        @param particle:
        @return area:
        """
        x: int = particle.x
        y: int = particle.y
        area: int = image[y: y + 31, x: x + 31].sum()

        return area

    def __get_total_count(self, image: np.ndarray, particles: list) -> int:
        """
        This method calculates the total area of all particles

        @param image:
        @param particles:
        @return total_area:
        """
        total_area: int = 0
        for p in particles:
            area = self.__calc_particle_area(image, p)
            total_area += area

        return total_area

    def __get_max_window(self, random_positions: list, image: np.array) -> tuple:
        """
        This method returns the particle that has the maximum area

        @param random_positions:
        @param image:
        @return x, y:
        """
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

    def __apply_morphological_operations(self, foreground: np.ndarray) -> np.ndarray:
        """
        This method applies 9 consecutive erode operations followed by 9 consecutive dilate operations
        with a 9x9 kernel

        @param foreground:
        @return foreground_dilated:
        """
        foreground = cv2.erode(foreground, self.__kernel, 9)
        return cv2.dilate(foreground, self.__kernel, 9)

    def __handle_zero_area(self, image, result):
        """
        This method handles the case when the total area of all particles is 0

        @param image:
        @param result:
        @return None:
        """

        self.__generate_particles()

        for p in self.__particles:
            cv2.rectangle(image, (p.x, p.y), (p.x + 31, p.y + 31),
                          (0, 255, 0), 1)

        cv2.imshow("All particles", np.uint8(image))
        cv2.imshow("Result", np.uint8(result))

    def __update_weights(self, foreground: np.ndarray, total_area: int) -> list:
        """
        This method updates the weights of all particles

        @param foreground:
        @param total_area:
        @return accumulated_weights:
        """
        aux: float = 0.0
        accumulated_weights: list = []
        for p in self.__particles:
            area = self.__calc_particle_area(foreground, p)

            p.weight = area / total_area

            aux += p.weight
            accumulated_weights.append(aux)

        return accumulated_weights

    def __difusion(self, p: Particle) -> Particle:
        """
        This method applies gaussian difusion to a particle

        @param p:
        @return particle:
        """
        dispersionx: int = int(p.x + 17 * np.random.normal(0, 1) + p.vx)
        dispersiony: int = int(p.y + 17 * np.random.normal(0, 1) + p.vy)

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

        return particle

    def __roulette_wheel_selection(self, accumulated_weights: list) -> list[Particle]:
        """
        This method applies roulette wheel selection to the particles

        @param accumulated_weights:
        @return new_particles:
        """
        new_particles: list[Particle] = []
        for i in range(len(self.__particles)):
            p: Particle = None
            rand_number: float = np.random.uniform(0, 1)
            for j in range(len(accumulated_weights)):
                if accumulated_weights[j] > rand_number:
                    p: Particle = self.__particles[j]
                    break

            if p is not None:
                # Difusion proccess
                particle: Particle = self.__difusion(p)

                new_particles.append(particle)

        return new_particles

    def __draw_particles(self, image: np.ndarray, particles: list[Particle]) -> None:
        """
        This method draws all particles in the image

        @param image:
        @param particles:
        @return None:
        """
        for p in particles:
            cv2.rectangle(image, (p.x, p.y), (p.x + 31, p.y + 31),
                          (0, 255, 0), 1)

    def track(self, img: str) -> bool:
        """
        This method tracks the object in the image

        @param img:
        @return bool:
        """
        foreground: np.ndarray = self.__substractor.substract(img)

        image: np.ndarray = cv2.imread(f"{PATH}{img}")
        result: np.ndarray = cv2.imread(f"{PATH}{img}")

        # Apply morphological operations
        foreground = self.__apply_morphological_operations(foreground)

        # Calculate the total area of all particles
        total_area: int = self.__get_total_count(foreground, self.__particles)

        if total_area == 0:
            self.__handle_zero_area(image, result)
            print("Total area is 0", end="\n\n")

        else:
            # Update weigths and create accumulated vector of weights
            accumulated_weights: list = self.__update_weights(foreground, total_area)

            # Estimate the position of the object with the particle with the maximum area
            max_window: tuple = self.__get_max_window([(p.x, p.y) for p in self.__particles], foreground)

            # Draw the particles
            self.__draw_particles(image, self.__particles)

            # Roulette wheel selection method
            new_particles: list = self.__roulette_wheel_selection(accumulated_weights)

            # Draw the rectangle of the estimated position in both images
            cv2.rectangle(image, (max_window[0], max_window[1]), (max_window[0] + 31, max_window[1] + 31), (0, 0, 255),
                          2)

            cv2.rectangle(result, (max_window[0], max_window[1]), (max_window[0] + 31, max_window[1] + 31), (0, 0, 255),
                          2)

            self.__particles = new_particles

        finished: bool = False

        cv2.imshow("All particles", np.uint8(image))
        cv2.imshow("Result", np.uint8(result))
        cv2.imshow("Foreground", np.uint8(foreground))

        if cv2.waitKey(300) & 0xFF == ord('q'):
            finished = True
            cv2.destroyAllWindows()

        return finished
