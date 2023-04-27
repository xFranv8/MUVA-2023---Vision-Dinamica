import cv2
import numpy as np


class BackSubstraction:
    def __init__(self, path: str):
        self.__path: str = path
        self.__background: np.array = np.array(cv2.imread(self.__path + "1.jpg"), dtype=np.float32)

    def substract(self, im2: str) -> np.array:
        """
        This method substracts the background from the foreground

        @param im2:
        @return foreground:
        """
        fg = cv2.imread(self.__path + im2)

        fg = np.array(fg, dtype=np.float32)

        diff = np.abs(fg - self.__background)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        diff = np.array(diff, dtype=np.uint8)
        thresh = np.array(thresh, dtype=np.uint8)

        return thresh