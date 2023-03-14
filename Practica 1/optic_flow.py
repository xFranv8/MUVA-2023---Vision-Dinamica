import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import time


def generate_image(size: (int, int)) -> (np.array, np.array):
    im1 = np.zeros(size, dtype=np.float32)
    im2 = np.zeros(size, dtype=np.float32)

    im1[160:240, 160:240] = 255
    im2[230:310, 230:310] = 255

    return np.array(im1), np.array(im2)


def calc_sum(img: np.matrix, size: int, x: int, y: int) -> float:
    mov = int(size / 2)
    # return np.power(img[155:155+size, 155:155+size], 2).sum()
    return np.power(img[y - mov:y + mov + 1, x - mov:x + mov + 1], 2).sum()


def lukas_kanade(gradx: np.matrix, grady: np.matrix, gradt: np.matrix, size: int, x: int, y: int) \
                -> (np.matrix, np.matrix, list):
    desp: int = size // 2

    sumx: float = calc_sum(gradx, size, x, y)
    sumy: float = calc_sum(grady, size, x, y)

    sumxy: float = (gradx[y - desp:y + desp + 1, x - desp:x + desp + 1] * grady[y - desp:y + desp + 1,
                                                                                  x - desp:x + desp + 1]) \
        .sum()

    sumxt: float = -(gradx[y - desp:y + desp + 1, x - desp:x + desp + 1] * gradt[y - desp:y + desp + 1,
                                                                                   x - desp:x + desp + 1]) \
        .sum()
    sumyt: float = -(grady[y - desp:y + desp + 1, x - desp:x + desp + 1] * gradt[y - desp:y + desp + 1,
                                                                                   x - desp:x + desp + 1]) \
        .sum()

    A: np.matrix = np.matrix([[sumx, sumxy], [sumxy, sumy]])
    b: np.matrix = np.matrix([[sumxt], [sumyt]])

    # Result with closed formula
    u = -(-sumy * sumxt + sumxy * sumyt) / (sumx * sumy - sumxy * sumxy)
    v = -(sumxy * sumxt - sumx * sumyt) / (sumx * sumy - sumxy * sumxy)

    return A, b, [u, v]


def horn_schunck(gradx: np.matrix, grady: np.matrix, gradt: np.matrix, landa: float, iterations: int,
                 shape: (int, int)) -> (np.array, np.array):

    u = np.zeros(gradx.shape)
    u_prev = np.ones(gradx.shape) * 1e-6

    v = np.zeros(gradx.shape)
    v_prev = np.ones(gradx.shape) * 1e-6

    avg = shape[0]*shape[1]
    avg_kernel = np.array([1/avg]*avg, float).reshape(shape)

    for it in range(iterations):
        aux = (gradx*u_prev + grady*v_prev + gradt)/(landa**2 + gradx**2 + grady**2)

        u = u_prev - gradx * aux
        v = v_prev - grady * aux

        u_prev = convolve(u, avg_kernel)
        v_prev = convolve(v, avg_kernel)

    return u, v


def draw_vector(im: np.array, point: (int, int), sol: (float, float), color: str) -> None:
    plt.imshow(im, cmap='gray')
    plt.quiver(point[0], point[1], sol[0], sol[1], angles='xy', color=color)


def run_lukas_kanade(gradx: np.array, grady: np.array, gradt: np.array, size: int, im: np.array, option: bool) -> None:
    for i in range(size//2, gradx.shape[1] - size//2 + 1):
        for j in range(size//2, gradx.shape[0] - size//2 + 1):
            A, b, sol = lukas_kanade(gradx, grady, gradt, size, j, i)
            if (i % 25) == 0 and (j % 25) == 0:
                """if option:
                    sol: np.array = np.array(np.linalg.pinv(A) @ b)
                else:
                    if np.linalg.det(A) != 0:
                        sol: np.array = np.array(np.linalg.inv(A) @ b)"""
                point = (j, i)
                color = "blue"
                if sol is not None and np.linalg.norm(sol) > 0.75:
                    draw_vector(im, point, sol, color)


def run_horn_schunck(gradx: np.array, grady: np.array, gradt: np.array, shape: (int, int), im: np.array) -> None:
    u, v = horn_schunck(gradx, grady, gradt, 10, 300, shape)
    for i in range(100, 400):
        for j in range(0, 400):
            if (i % 25) == 0 and (j % 25) == 0:
                if np.linalg.norm((v[i, j], u[i, j])) > 0.5:
                    print(f"Norma: {np.linalg.norm((v[i, j], u[i, j]))} \n Vector: {u[i, j]}, {v[i, j]}")
                    draw_vector(im, (j, i), (u[i, j], v[i, j]), "red")


def main():
    # im1, im2 = generate_image((400, 400))

    im1 = np.float32(cv2.imread("images/car1.jpg"))
    im2 = np.float32(cv2.imread("images/car2.jpg"))

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1 = np.float32(cv2.GaussianBlur(im1, (5, 5), 0))
    im2 = np.float32(cv2.GaussianBlur(im2, (5, 5), 0))

    gradx1: np.array = cv2.Sobel(im1, cv2.CV_64F, dx=1, dy=0, ksize=3)
    grady1: np.array = cv2.Sobel(im1, cv2.CV_64F, dx=0, dy=1, ksize=3)
    gradx2: np.array = cv2.Sobel(im2, cv2.CV_64F, dx=1, dy=0, ksize=3)
    grady2: np.array = cv2.Sobel(im2, cv2.CV_64F, dx=0, dy=1, ksize=3)

    gradx: np.array = 0.5 * (gradx1 + gradx2)
    grady: np.array = 0.5 * (grady1 + grady2)

    gradt: np.array = np.float32(im2 - im1)

    # LUKAS-KANADE CON VENTANA 3x3 pinv
    """print("Running Lukas-Kanade with formula...")

    size: int = 37
    option: bool = True

    t0 = time.time()
    run_lukas_kanade(gradx, grady, gradt, size, im2, option)
    t1 = time.time() - t0

    print(f"Time for LK formula 3x3: {t1}")"""

    # HORN-SCHUNCK con ventana 7x7
    print("Running Horn-Schunck...")
    
    t0 = time.time()
    run_horn_schunck(gradx, grady, gradt, (5, 5), im2)
    t1 = time.time() - t0
    
    print(f"Time for HS 3x3: {t1}")

    plt.show()


if __name__ == '__main__':
    main()
