import os
import numpy as np
import cv2
import threading
import time
import torch
import torch.nn.functional as F

# Загрузка изображения в градациях серого
# Нормировка к диапазону [0, 1]


def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img.astype(np.float32) / 255.0

# Нормализация ядра (PSF), чтобы сумма элементов была 1
# ∑ k = 1


def normalize_kernel(kernel):
    kernel_sum = np.sum(kernel)
    return kernel / kernel_sum if kernel_sum != 0 else kernel

# Расчет PSNR (Peak Signal-to-Noise Ratio)
# PSNR = 20 * log10(MAX / sqrt(MSE))


def psnr(target, ref):
    mse = torch.mean((target - ref) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))

# Суррогат TV-регуляризации (Total Variation)
# TV(x) ≈ ∑√((∂x/∂x)² + (∂x/∂y)² + ε)
# Сглаженная вариация для устойчивого градиента
# Используется для минимизации градиента на границах и устранения шумов


def total_variation_surrogate(x, eps=1e-3):
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    tv = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2 + eps).sum()
    return tv

# Экспоненциальное затухание параметра регуляризации
# λ(i) = λ₀ * decay^i, λ ≥ min_lambda
# Используется для плавного уменьшения влияния регуляризации в процессе итераций


def exponential_decay_lambda(i, lambda_start=0.01, decay=0.995, min_lambda=1e-5):
    return max(lambda_start * (decay ** i), min_lambda)

# Основная функция деконволюции с использованием TV или Tikhonov регуляризации
# Решает задачу восстановления x из y в задаче типа Kx ≈ y
# с функционалом:
# E(x) = ||Kx - y||² + λ * R(x)
# где R(x) = TV(x) или ||x||²₂ (Тихонов)


def deconvolve_pytorch(y_np, k_np, method="TV", num_iters=300, lambda_reg=0.001, alpha=0.5, gamma=0.8, tol=1e-4, device='cuda', tv_interval=1, result_dict=None, key=None):
    y = torch.from_numpy(y_np).to(device).unsqueeze(0).unsqueeze(0)
    k = torch.from_numpy(k_np).to(device).unsqueeze(0).unsqueeze(0)
    k_flip = torch.flip(k, [2, 3])  # Отражение ядра для обратной свертки

    x = y.clone().detach().requires_grad_(True)  # Инициализация x
    v = torch.zeros_like(x)  # Импульс (momentum)
    prev_x = x.clone()

    start = time.time()

    for i in range(num_iters):
        # Шаг 1: вычисление градиента ошибки восстановления
        conv_x = F.conv2d(x, k, padding='same')
        residual = conv_x - y
        grad_data = F.conv2d(
            residual, k_flip, padding='same')  # ∇(||Kx - y||²)

        # Шаг 2: обновление импульса и переменной x (с градиентным шагом)
        v = gamma * v + alpha * grad_data
        x = x - v
        x = torch.clamp(x, 0.0, 1.0)

        # Шаг 3: вычисление адаптивного lambda
        adaptive_lambda = exponential_decay_lambda(i, lambda_start=lambda_reg)

        # Шаг 4: добавление регуляризации в зависимости от метода
        if method == "TV" and i % tv_interval == 0:
            # ∇TV(x), считаем градиент через автодифференцирование
            grad_tv = torch.autograd.grad(
                total_variation_surrogate(x), x, create_graph=False)[0]
            x = x - alpha * adaptive_lambda * grad_tv
            x = torch.clamp(x, 0.0, 1.0)
        elif method == "Tikhonov":
            # ∇(||x||²₂) = 2x — градиент функционала нормы L2 (см. Тихонов, Арсенин, Гл. II)
            grad_tik = 2 * x
            x = x - alpha * adaptive_lambda * grad_tik
            x = torch.clamp(x, 0.0, 1.0)

        # Критерий сходимости
        diff = torch.norm(x - prev_x) / (torch.norm(prev_x) + 1e-8)
        print(
            f"{method} итерация {i+1}, Δ = {diff.item():.6f}, lambda = {adaptive_lambda:.6f}")
        if diff < tol:
            print(f"{method}: Сходимость достигнута на итерации {i+1}.")
            break

        prev_x = x.clone()

    end = time.time()

    result = (x.detach().squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    if result_dict is not None and key is not None:
        result_dict[key] = (result, end - start)
    return result

# Главная функция запуска
# Загружает входное изображение и PSF, запускает оба метода и сохраняет результат


def main():
    input_file = "blurred.bmp"  # Входное размытое изображение
    kernel_file = "kernel.bmp"  # Ядро (PSF)
    output_tv = "output_tv.png"
    output_tikhonov = "output_tikhonov.png"

    if not os.path.exists(input_file) or not os.path.exists(kernel_file):
        raise FileNotFoundError("Файл изображения или ядра не найден")

    y = load_gray_image(input_file)
    k = load_gray_image(kernel_file)
    k = normalize_kernel(k)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")

    results = {}

    # Параллельный запуск двух методов
    t1 = threading.Thread(target=deconvolve_pytorch, args=(y, k), kwargs={
        'method': 'TV',
        'num_iters': 800,
        'lambda_reg': 0.01,
        'tol': 1e-3,
        'device': device,
        'result_dict': results,
        'key': 'TV'
    })

    t2 = threading.Thread(target=deconvolve_pytorch, args=(y, k), kwargs={
        'method': 'Tikhonov',
        'num_iters': 800,
        'lambda_reg': 0.01,
        'tol': 1e-3,
        'device': device,
        'result_dict': results,
        'key': 'Tikhonov'
    })

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Сохранение результатов
    cv2.imwrite(output_tv, results['TV'][0])
    cv2.imwrite(output_tikhonov, results['Tikhonov'][0])

    print(f"TV регуляризация: Время = {results['TV'][1]:.2f} сек")
    print(f"Тихонов регуляризация: Время = {results['Tikhonov'][1]:.2f} сек")


if __name__ == "__main__":
    main()
