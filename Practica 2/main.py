import natsort
import os
from ParticleFilter import ParticleFilter


def main():
    PATH: str = "Secuencia/"

    tracker = ParticleFilter(PATH)

    images: list = natsort.natsorted(os.listdir(PATH))
    for img in images:
        finished = tracker.track(img)
        if finished:
            break


if __name__ == '__main__':
    main()
