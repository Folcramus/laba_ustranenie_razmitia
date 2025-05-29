import os
import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage.restoration import denoise_tv_chambolle


def load_gray_image(path):
    """
    Загрузка изображения в градациях серого и преобразование к типу float64
    Args:
        path: Путь к файлу изображения
    Returns:
        Изображение в градациях серого (float64)
    Raises:
        ValueError: Если изображение не удалось загрузить
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Чтение в оттенках серого
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    # Конвертация в float64 для точных вычислений
    return img.astype(np.float64)


def normalize_kernel(kernel):
    """
    Нормализация ядра размытия - сумма значений должна быть равна 1
    Args:
        kernel: Ядро размытия (матрица)
    Returns:
        Нормализованное ядро
    """
    kernel_sum = np.sum(kernel)
    # Деление на сумму или исходное ядро
    return kernel / kernel_sum if kernel_sum != 0 else kernel


def psnr(target, ref):
    """
    Вычисление Peak Signal-to-Noise Ratio (PSNR) между двумя изображениями
    Args:
        target: Восстановленное изображение
        ref: Эталонное изображение
    Returns:
        Значение PSNR в децибелах (dB)
    """
    mse = np.mean((target - ref) ** 2)  # Среднеквадратичная ошибка
    if mse == 0:
        return float("inf")  # Бесконечность если изображения идентичны
    PIXEL_MAX = 255.0  # Максимальное значение пикселя
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))  # Формула PSNR


def deconvolve_tv(y, k, num_iters=100, lambda_tv=0.1):
    """
    Метод деконволюции с Total Variation (TV) регуляризацией
    Args:
        y: Размытое изображение (наблюдение)
        k: Ядро размытия
        num_iters: Количество итераций (по умолчанию 100)
        lambda_tv: Параметр регуляризации (по умолчанию 0.1)
    Returns:
        Восстановленное изображение
    """
    # Отражение ядра по вертикали и горизонтали необходимо для корректного
    # вычисления градиента (эквивалент операции транспонирования)
    # Это нужно потому, что свертка с отраженным ядром - это сопряжённый оператор
    k_flip = np.flipud(np.fliplr(k))

    # В качестве начального приближения берем само размытое изображение
    x = y.copy()

    for i in range(num_iters):
        print(f"Итерация {i+1}/{num_iters}")

        # 1.1. Прямая операция (имитация размытия)
        # -------------------------------------------------
        # Вычисляем, как выглядело бы текущее восстановленное изображение x,
        # если бы его размыли ядром k
        # boundary='symm' - симметричное продолжение границ для уменьшения артефактов
        conv_x = convolve2d(x, k, mode='same', boundary='symm')

        # 1.2. Вычисление невязки (ошибки реконструкции)
        # -------------------------------------------------
        # Разница между смоделированным размытием и реальным размытым изображением

        residual = conv_x - y

        # 1.3. Обратное распространение ошибки
        # -------------------------------------------------
        # Вычисление градиента функции потерь относительно x
        # Эквивалентно умножению на сопряженный оператор (k^T)
        grad_data = convolve2d(residual, k_flip, mode='same', boundary='symm')

        # 1.4. Градиентный спуск
        # -------------------------------------------------
        # Обновление текущей оценки в направлении, уменьшающем ошибку
        # Изменить на метод Нестерова
        x = x - 0.1 * grad_data

        x_normalized = x / 255.0
        # TV-регуляризация решает две задачи:
        # 1. Подавление шумов и артефактов
        # 2. Сохранение резких границ

        # Нормализация в [0, 1] (требование denoise_tv_chambolle)
        # Посмотреть что это за функция и как она работает
        x_denoised = denoise_tv_chambolle(x_normalized, weight=lambda_tv)

        # Возвращаем значения в исходный диапазон [0, 255]
        # и обеспечиваем корректные значения пикселей
        x = np.clip(x_denoised * 255.0, 0, 255)

    return x


def main():
    """
    Основная функция: загрузка данных, деконволюция, сохранение результата
    """
    # Параметры по умолчанию
    input_file = "Test6.png"    # Размытое изображение
    kernel_file = "psf_kernel_6.png"    # Ядро размытия
    output_file = "output.bmp"    # Выходной файл
    noise_level = 0.0001        # Уровень шума (влияет на силу регуляризации)

    # Проверка наличия файлов
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Файл {input_file} не найден")
    if not os.path.exists(kernel_file):
        raise FileNotFoundError(f"Файл {kernel_file} не найден")

    # Загрузка данных
    y = load_gray_image(input_file)  # Загрузка размытого изображения
    k = load_gray_image(kernel_file)  # Загрузка ядра размытия
    k = normalize_kernel(k)         # Нормализация ядра

    # Деконволюция с TV-регуляризацией
    print("Начало процесса деконволюции...")
    x_restored = deconvolve_tv(
        y, k,
        num_iters= 800,               # Количество итераций
        lambda_tv=(noise_level / 255)  # Параметр регуляризации
    )
    print("Деконволюция завершена!")

    # Сохранение результата
    cv2.imwrite(output_file, np.uint8(np.clip(x_restored, 0, 255)))
    print(f"Результат сохранен в {output_file}")


if __name__ == "__main__":
    main()  # Точка входа в программу
