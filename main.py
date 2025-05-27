import os
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_chambolle


def load_gray_image(path):
    """Загрузка изображения в градациях серого и преобразование к типу float64"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img.astype(np.float64)


def normalize_kernel(kernel):
    """Нормализация ядра: сумма значений должна быть равна 1"""
    kernel_sum = np.sum(kernel)
    return kernel / kernel_sum if kernel_sum != 0 else kernel


def psnr(target, ref):
    """Подсчёт PSNR"""
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def deconvolve_tv(y, k, num_iters=100, lambda_tv=0.1):
    """Метод деконволюции с регуляризацией по полной вариации (TV)"""
    k_flip = np.flipud(np.fliplr(k))  # Отражение ядра
    x = y.copy()  # Инициализация

    for i in range(num_iters):
        print(i)
        # Считаем градиент невязки: k^T * (k * x - y)
        conv_x = convolve2d(x, k, mode='same', boundary='symm')
        residual = conv_x - y
        grad_data = convolve2d(residual, k_flip, mode='same', boundary='symm')

        # Обновляем x с шагом градиентного спуска
        x = x - 0.1 * grad_data  # шаг можно адаптировать

        # Применяем регуляризацию TV
        x = denoise_tv_chambolle(x / 255.0, weight=lambda_tv)
        x = np.clip(x * 255.0, 0, 255)

    return x


def main():
    # Имена файлов по умолчанию
    input_file = "blurred.bmp"
    kernel_file = "kernel.bmp"
    output_file = "output.bmp"
    noise_level = 0.00001  # Значение по умолчанию

    # Проверяем наличие файлов в текущей директории
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Файл {input_file} не найден в текущей директории")

    if not os.path.exists(kernel_file):
        raise FileNotFoundError(f"Файл {kernel_file} не найден в текущей директории")

    # Загрузка изображений
    y = load_gray_image(input_file)
    k = load_gray_image(kernel_file)
    k = normalize_kernel(k)

    # Выполнение деконволюции
    x_restored = deconvolve_tv(
        y, k, num_iters=200, lambda_tv=(noise_level / 255))

    # Сохранение результата
    cv2.imwrite(output_file, np.uint8(np.clip(x_restored, 0, 255)))
    print(f"Результат сохранен в файл {output_file}")


if __name__ == "__main__":
    main()