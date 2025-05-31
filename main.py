import os
import numpy as np
import cv2
import threading
import time
import torch
import torch.nn.functional as F


def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {path}")
    return img.astype(np.float32) / 255.0


def normalize_kernel(kernel):
    kernel_sum = np.sum(kernel)
    return kernel / kernel_sum if kernel_sum != 0 else kernel


def psnr(target, ref):
    mse = torch.mean((target - ref) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))


def total_variation(x):
    # TV регуляризация: сумма разностей между соседними пикселями по горизонтали и вертикали
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return tv_h + tv_w


def tikhonov_regularization(x):
    # Регуляризация Тихонова: ||x||^2 = сумма квадратов значений
    return torch.sum(x ** 2)


def deconvolve_pytorch(y_np, k_np, method="TV", num_iters=300, lambda_reg=0.001, alpha=0.5, gamma=0.9, tol=1e-4, device='cuda', tv_interval=3, result_dict=None, key=None):
    # Перевод изображений и ядра в тензоры PyTorch
    y = torch.from_numpy(y_np).to(device).unsqueeze(0).unsqueeze(0)
    k = torch.from_numpy(k_np).to(device).unsqueeze(0).unsqueeze(0)
    k_flip = torch.flip(k, [2, 3])

    x = y.clone().detach().requires_grad_(True)
    v = torch.zeros_like(x)
    prev_x = x.clone()

    start = time.time()

    for i in range(num_iters):
        # Шаг восстановления с помощью свёртки и градиентного спуска
        conv_x = F.conv2d(x, k, padding='same')
        residual = conv_x - y
        grad_data = F.conv2d(residual, k_flip, padding='same')

        # Адаптивный шаг градиентного спуска (метод Нестерова)
        v = gamma * v + alpha * grad_data
        x = x - v
        x = torch.clamp(x, 0.0, 1.0)

        # Применение регуляризации в зависимости от метода
        if method == "TV" and i % tv_interval == 0:
            x = x - alpha * lambda_reg * \
                torch.autograd.grad(total_variation(x), x,
                                    create_graph=False)[0]
            x = torch.clamp(x, 0.0, 1.0)
        elif method == "Tikhonov":
            x = x - alpha * lambda_reg * \
                torch.autograd.grad(tikhonov_regularization(
                    x), x, create_graph=False)[0]
            x = torch.clamp(x, 0.0, 1.0)

        # Проверка на сходимость по относительной норме разности
        diff = torch.norm(x - prev_x) / (torch.norm(prev_x) + 1e-8)
        print(f"{method} итерация {i+1}, Δ = {diff.item():.6f}")
        if diff < tol:
            print(f"{method}: Сходимость достигнута на итерации {i+1}.")
            break

        prev_x = x.clone()

    end = time.time()

    result = (x.detach().squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    if result_dict is not None and key is not None:
        result_dict[key] = (result, end - start)
    return result


def main():
    input_file = "Test6.png"
    kernel_file = "psf_kernel_6_test.png"
    output_tv = "output_tv.bmp"
    output_tikhonov = "output_tikhonov.bmp"
    noise_level = 0.0001

    if not os.path.exists(input_file) or not os.path.exists(kernel_file):
        raise FileNotFoundError("Файл изображения или ядра не найден")

    y = load_gray_image(input_file)
    k = load_gray_image(kernel_file)
    k = normalize_kernel(k)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем устройство: {device}")

    results = {}

    # Запуск двух потоков для сравнения методов TV и Тихонова
    t1 = threading.Thread(target=deconvolve_pytorch, args=(y, k), kwargs={
        'method': 'TV',
        'num_iters': 800,
        'lambda_reg': noise_level,
        'tol': 2e-3,
        'device': device,
        'result_dict': results,
        'key': 'TV'
    })

    t2 = threading.Thread(target=deconvolve_pytorch, args=(y, k), kwargs={
        'method': 'Tikhonov',
        'num_iters': 800,
        'lambda_reg': noise_level,
        'tol': 2e-3,
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

    # Вывод времени выполнения
    print(f"TV регуляризация: Время = {results['TV'][1]:.2f} сек")
    print(f"Тихонов регуляризация: Время = {results['Tikhonov'][1]:.2f} сек")


if __name__ == "__main__":
    main()
