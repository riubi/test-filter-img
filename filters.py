import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "images/chelsea.png"
OUT = "results"
SIZE = 256

np.random.seed(0)


def load_image(path, size=SIZE):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def add_gauss_noise(img, sigma):
    noisy = img.astype(np.float64) + np.random.normal(0, sigma, img.shape)
    return to_u8(noisy)


def add_salt_pepper(img, p):
    noisy = img.copy()
    r = np.random.rand(*img.shape[:2])
    noisy[r < p / 2] = 0
    noisy[r > 1 - p / 2] = 255
    return noisy


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def conv2d(img, kernel, stride=1):
    img = img.astype(np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if img.ndim == 3:
        return np.dstack([conv2d(img[:, :, c], kernel, stride) for c in range(img.shape[2])])

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    h, w = img.shape
    out = np.zeros(((h + stride - 1) // stride, (w + stride - 1) // stride))
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h:stride, j:j + w:stride]
    return out


def sobel_kernels(ksize):
    kx, ky = cv2.getDerivKernels(1, 0, ksize)
    gx = np.outer(ky.ravel(), kx.ravel())
    gx = gx / gx[gx > 0].sum()
    return gx, gx.T.copy()


PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float64) / 3

SHARPEN_3 = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float64)

HIGHPASS_3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float64)


def gauss_kernel(ksize, sigma=0, sigma_y=None):
    if sigma <= 0:
        sigma = default_sigma(ksize)
    if sigma_y is None:
        sigma_y = sigma
    gx = cv2.getGaussianKernel(ksize, sigma).ravel()
    gy = cv2.getGaussianKernel(ksize, sigma_y).ravel()
    return np.outer(gy, gx)


def auto_ksize(sigma):
    return 2 * int(np.ceil(3 * sigma)) + 1


def default_sigma(ksize):
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def sharpen_kernel(ksize, amount, sigma=0):
    g = gauss_kernel(ksize, sigma)
    delta = np.zeros_like(g)
    delta[ksize // 2, ksize // 2] = 1
    return delta + amount * (delta - g)


def sobel(gray, ksize=3, stride=1, pre_sigma=0):
    src = gray.astype(np.float64)
    if pre_sigma > 0:
        src = cv2.GaussianBlur(src, (0, 0), pre_sigma)
    kx, ky = sobel_kernels(ksize)
    gx = conv2d(src, kx, stride)
    gy = conv2d(src, ky, stride)
    return gx, gy


def magnitude(gx, gy, kind="L2"):
    if kind == "L1":
        return np.abs(gx) + np.abs(gy)
    return np.sqrt(gx ** 2 + gy ** 2)


def unsharp(img, sigma, amount, threshold=0):
    img = img.astype(np.float64)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - blur
    if img.ndim == 3:
        mask = (np.abs(detail.mean(axis=2)) >= threshold)[:, :, None]
    else:
        mask = np.abs(detail) >= threshold
    return img + amount * detail * mask


def highpass_fft(gray, d0):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    h, w = gray.shape
    v, u = np.mgrid[-h // 2:h - h // 2, -w // 2:w - w // 2]
    d2 = u ** 2 + v ** 2
    H = 1 - np.exp(-d2 / (2 * d0 ** 2))
    return np.real(np.fft.ifft2(np.fft.ifftshift(f * H)))


def save_grid(images, titles, ncols, fname, suptitle=None, cell=2.6):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    lines = max(t.count("\n") + 1 for t in titles)
    height = (cell + 0.25 * lines + 0.1) * nrows + (0.5 if suptitle else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, height), layout="constrained")
    fs = min(11.5, max(8.5, 1.05 * cell * ncols))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, images, titles):
        im = to_u8(im)
        if im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.set_title(t, fontsize=fs)
    if suptitle:
        fig.suptitle(suptitle, fontsize=fs + 2)
    fig.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close(fig)


def crop(img, box):
    y, x, s = box
    return img[y:y + s, x:x + s]


def edge_view(x, gain):
    return np.abs(x) * gain


def hp_view(x):
    return x + 128


def check_conv(gray, img):
    rnd = np.random.rand(7, 7)
    ref = cv2.filter2D(gray.astype(np.float64), -1, rnd, borderType=cv2.BORDER_REFLECT_101)
    mine = conv2d(gray, rnd)
    print("conv2d vs filter2D, max diff:", np.abs(ref - mine).max())
    assert np.allclose(ref, mine)

    for k in (3, 7):
        ref = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=k)
        kx, _ = sobel_kernels(k)
        raw = cv2.getDerivKernels(1, 0, k)
        norm = np.outer(raw[1].ravel(), raw[0].ravel())
        norm = norm[norm > 0].sum()
        mine = conv2d(gray, kx)
        print(f"sobel {k}x{k} vs cv2.Sobel, max diff:", np.abs(ref / norm - mine).max())
        assert np.allclose(ref / norm, mine, atol=1e-6)

        ref = cv2.GaussianBlur(img.astype(np.float64), (k, k), default_sigma(k))
        mine = conv2d(img, gauss_kernel(k))
        print(f"gauss {k}x{k} vs cv2.GaussianBlur, max diff:", np.abs(ref - mine).max())
        assert np.allclose(ref, mine)

    full = conv2d(gray, rnd)
    for s in (2, 3):
        out = conv2d(gray, rnd, s)
        assert out.shape == full[::s, ::s].shape
        assert np.allclose(out, full[::s, ::s])
        print(f"stride={s}: shape {out.shape}, совпадает с full[::{s}, ::{s}]")


def part_original(orig, img, gray):
    save_grid([orig, img, gray],
              [f"Исходное {orig.shape[1]}x{orig.shape[0]}",
               f"Обрезка + resize {img.shape[1]}x{img.shape[0]}",
               "Оттенки серого 256x256"],
              3, "01_original.png", cell=3.0)


def part_edges_all(gray):
    g = gray.astype(np.float64)
    gain = 3
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) / 4
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) / 4
    px = conv2d(g, PREWITT_X)
    py = conv2d(g, PREWITT_X.T)
    shx = cv2.Scharr(g, cv2.CV_64F, 1, 0) / 16
    shy = cv2.Scharr(g, cv2.CV_64F, 0, 1) / 16
    lap = cv2.Laplacian(g, cv2.CV_64F, ksize=1)

    images = [gray,
              edge_view(sx, gain), edge_view(sy, gain), edge_view(magnitude(sx, sy), gain),
              edge_view(px, gain), edge_view(py, gain), edge_view(magnitude(shx, shy), gain),
              edge_view(lap, gain)]
    titles = ["Исходное (gray)",
              "Sobel X, 3x3", "Sobel Y, 3x3", "Sobel X+Y, 3x3\nsqrt(Gx²+Gy²)",
              "Prewitt X, 3x3", "Prewitt Y, 3x3", "Scharr X+Y, 3x3",
              "Laplacian, 3x3\n[0 1 0; 1 -4 1; 0 1 0]"]
    save_grid(images, titles, 4, "02_edges_all.png",
              suptitle=f"Фильтры выделения границ (|отклик| x {gain}, ядра нормированы)")


def part_smoothing_all(img):
    k = 7
    noisy_g = add_gauss_noise(img, 20)
    noisy_sp = add_salt_pepper(img, 0.05)
    images, titles = [], []
    for src, name in [(img, "без шума"), (noisy_g, "гаусс. шум σ=20"), (noisy_sp, "соль-перец 5%")]:
        images += [src,
                   cv2.blur(src, (k, k)),
                   cv2.GaussianBlur(src, (k, k), default_sigma(k)),
                   cv2.medianBlur(src, k),
                   cv2.bilateralFilter(src, k, 75, 75)]
        titles += [f"Вход: {name}",
                   f"Box Blur {k}x{k}",
                   f"Gaussian {k}x{k}, σ={default_sigma(k):.1f}",
                   f"Median {k}x{k}",
                   f"Bilateral d={k}, σc=75, σs=75"]
    save_grid(images, titles, 5, "03_smoothing_all.png",
              suptitle="Фильтры сглаживания на чистом и зашумлённых изображениях")


def part_sharpen_all(img, gray):
    g = gray.astype(np.float64)
    images = [img,
              conv2d(img, SHARPEN_3),
              hp_view(conv2d(g, HIGHPASS_3)),
              hp_view(g - cv2.GaussianBlur(g, (0, 0), 2)),
              hp_view(highpass_fft(g, 10)),
              hp_view(highpass_fft(g, 30)),
              hp_view(highpass_fft(g, 60)),
              conv2d(img, HIGHPASS_3) + img.astype(np.float64)]
    titles = ["Исходное",
              "Sharpening 3x3\n[0 -1 0; -1 5 -1; 0 -1 0]",
              "High-pass 3x3\n[-1..8..-1] (+128)",
              "High-pass\nI - Gauss(σ=2) (+128)",
              "High-pass FFT\nD0=10 (+128)",
              "High-pass FFT\nD0=30 (+128)",
              "High-pass FFT\nD0=60 (+128)",
              "I + High-pass 3x3\n(ядро с 9 в центре)"]
    save_grid(images, titles, 4, "04_sharpen_all.png",
              suptitle="Фильтры повышения резкости и высокочастотные фильтры")


def part_sobel(gray):
    gain = 3
    images, titles = [], []
    for k in (3, 7):
        for s in (1, 2, 3):
            gx, gy = sobel(gray, k, s)
            m = magnitude(gx, gy)
            images.append(edge_view(m, gain))
            titles.append(f"Sobel X+Y, {k}x{k}, stride={s}\nвыход {m.shape[1]}x{m.shape[0]}")
    save_grid(images, titles, 3, "05_sobel_window_stride.png", cell=3.0,
              suptitle="Sobel X+Y: размер окна и шаг (|G| x 3)")

    gx, gy = sobel(gray, 3)
    images = [edge_view(gx, gain), edge_view(gy, gain)]
    titles = ["θ=0° (Sobel X)", "θ=90° (Sobel Y)"]
    for t in (45, 135):
        r = np.deg2rad(t)
        images.append(edge_view(gx * np.cos(r) + gy * np.sin(r), gain))
        titles.append(f"θ={t}°: Gx·cosθ + Gy·sinθ")
    images += [edge_view(magnitude(gx, gy, "L2"), gain), edge_view(magnitude(gx, gy, "L1"), gain)]
    titles += ["|G| = sqrt(Gx²+Gy²) (L2)", "|G| = |Gx|+|Gy| (L1)"]
    save_grid(images, titles, 3, "06_sobel_direction.png", cell=3.0,
              suptitle="Sobel 3x3: направление производной и способ объединения X и Y (|отклик| x 3)")

    images, titles = [], []
    for ps in (0, 1, 2):
        for k in (3, 5, 7):
            gx, gy = sobel(gray, k, pre_sigma=ps)
            images.append(edge_view(magnitude(gx, gy), gain))
            titles.append(f"ksize={k}, pre-blur σ={ps}")
    save_grid(images, titles, 3, "07_sobel_ksize_blur.png", cell=3.0,
              suptitle="Sobel X+Y: размер ядра и гауссово размытие перед фильтром (|G| x 3)")

    gx, gy = sobel(gray, 3, pre_sigma=1)
    m = magnitude(gx, gy)
    images = [edge_view(m, gain)]
    titles = ["|G|, 3x3, pre-blur σ=1"]
    for t in (10, 20, 40):
        images.append((m > t) * 255.0)
        pct = (m > t).mean() * 100
        titles.append(f"|G| > {t}  ({pct:.1f}% пикс.)")
    save_grid(images, titles, 4, "08_sobel_threshold.png",
              suptitle="Sobel X+Y: порог бинаризации")


def part_gauss(img):
    images, titles = [], []
    for k in (3, 7):
        for s in (1, 2, 3):
            out = conv2d(img, gauss_kernel(k), s)
            images.append(out)
            titles.append(f"Gaussian {k}x{k} (σ={default_sigma(k):.1f}), stride={s}\nвыход {out.shape[1]}x{out.shape[0]}")
    save_grid(images, titles, 3, "09_gauss_window_stride.png", cell=3.0,
              suptitle="Gaussian Blur: размер окна и шаг")

    images = [img]
    titles = ["Исходное"]
    for k in (3, 7):
        images.append(conv2d(img, gauss_kernel(k, 1.0)))
        titles.append(f"Gaussian {k}x{k}, σ=1")
    for k in (3, 7):
        images.append(conv2d(img, gauss_kernel(k, 3.0)))
        titles.append(f"Gaussian {k}x{k}, σ=3")
    images.append(conv2d(img, gauss_kernel(auto_ksize(3.0), 3.0)))
    titles.append(f"Gaussian {auto_ksize(3.0)}x{auto_ksize(3.0)}, σ=3")
    save_grid(images, titles, 3, "10_gauss_window_same_sigma.png", cell=3.0,
              suptitle="Gaussian Blur: одна и та же σ при разных окнах")

    images, titles = [], []
    for sg in (0.5, 1, 2, 4):
        k = auto_ksize(sg)
        images.append(conv2d(img, gauss_kernel(k, sg)))
        titles.append(f"σ={sg}, окно {k}x{k}")
    k = auto_ksize(4)
    images.append(conv2d(img, gauss_kernel(k, 4, 0.5)))
    titles.append(f"σx=4, σy=0.5, окно {k}x{k}")
    images.append(conv2d(img, gauss_kernel(k, 0.5, 4)))
    titles.append(f"σx=0.5, σy=4, окно {k}x{k}")
    save_grid(images, titles, 3, "11_gauss_sigma.png", cell=3.0,
              suptitle="Gaussian Blur: стандартное отклонение σ и разные σ по осям")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for sg in (0.5, 1, 2, 4):
        g = cv2.getGaussianKernel(25, sg).ravel()
        ax.plot(np.arange(-12, 13), g, marker=".", label=f"σ={sg}")
    ax.axvspan(-3.5, 3.5, color="gray", alpha=0.12, label="окно 7x7")
    ax.axvspan(-1.5, 1.5, color="tab:orange", alpha=0.15, label="окно 3x3")
    ax.set_xlabel("смещение от центра, пикс.")
    ax.set_ylabel("вес")
    ax.set_title("Одномерный профиль ядра Гаусса при разных σ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "12_gauss_profiles.png"), dpi=200)
    plt.close(fig)

    noisy = add_gauss_noise(img, 20)
    images = [noisy]
    titles = [f"Шум σn=20, PSNR={psnr(noisy, img):.1f} дБ"]
    for sg in (0.5, 1, 1.5, 2, 3):
        k = auto_ksize(sg)
        out = to_u8(conv2d(noisy, gauss_kernel(k, sg)))
        images.append(out)
        titles.append(f"σ={sg} ({k}x{k}), PSNR={psnr(out, img):.1f} дБ")
    save_grid(images, titles, 3, "13_gauss_noise.png", cell=3.0,
              suptitle="Gaussian Blur на зашумлённом изображении (PSNR относительно чистого)")


def part_sharpen(img):
    box = (56, 40, 88)
    amount = 1.5

    images, titles = [], []
    for k in (3, 7):
        for s in (1, 2, 3):
            out = conv2d(img, sharpen_kernel(k, amount), s)
            images.append(out)
            titles.append(f"Sharpen {k}x{k}, k={amount}, σ={default_sigma(k):.1f}, stride={s}\nвыход {out.shape[1]}x{out.shape[0]}")
    save_grid(images, titles, 3, "14_sharp_window_stride.png", cell=3.0,
              suptitle="Sharpening (unsharp mask): размер окна и шаг")

    images = [crop(img, box)]
    titles = ["Исходное (фрагмент)"]
    for k in (3, 7):
        images.append(crop(conv2d(img, sharpen_kernel(k, amount)), box))
        titles.append(f"Sharpen {k}x{k}, k={amount} (фрагмент)")
    save_grid(images, titles, 3, "15_sharp_window_crop.png", cell=3.0)

    images = [img]
    titles = ["Исходное"]
    for a in (0.5, 1, 2, 4):
        images.append(unsharp(img, 1.0, a))
        titles.append(f"k={a}, σ=1")
    images += [crop(im, box) for im in images]
    titles += titles
    save_grid(images, titles, 5, "16_sharp_amount.png", cell=2.6,
              suptitle="Sharpening: коэффициент усиления k (I + k·(I - G)), внизу увеличенный фрагмент")

    images = [img]
    titles = ["Исходное"]
    for sg in (0.5, 1, 2, 4):
        images.append(unsharp(img, sg, amount))
        titles.append(f"σ={sg}, k={amount}")
    images += [crop(im, box) for im in images]
    titles += titles
    save_grid(images, titles, 5, "17_sharp_radius.png", cell=2.6,
              suptitle="Sharpening: радиус σ гауссова размытия, внизу увеличенный фрагмент")

    noisy = add_gauss_noise(img, 6)
    images = [noisy]
    titles = ["Вход: слабый шум σn=6"]
    for t in (0, 5, 10, 20):
        images.append(unsharp(noisy, 1.0, 2.0, t))
        titles.append(f"k=2, σ=1, порог={t}")
    images += [crop(im, box) for im in images]
    titles += titles
    save_grid(images, titles, 5, "18_sharp_threshold.png", cell=2.6,
              suptitle="Sharpening: порог (усиливаются только перепады больше порога), внизу фрагмент")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    row = 100
    x = np.arange(30, 136)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(x, gray[row, x], color="black", lw=2, label="исходное")
    for a in (1, 4):
        sh = np.clip(unsharp(gray, 1.0, a), 0, 255)
        ax.plot(x, sh[row, x], lw=1, label=f"k={a}, σ=1")
    ax.set_xlabel("x, пикс.")
    ax.set_ylabel("яркость")
    ax.set_title(f"Профиль яркости вдоль строки y={row}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "19_sharp_profile.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    orig = cv2.imread(SRC)
    img = load_image(SRC)
    cv2.imwrite("images/input_256.png", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("исходный размер:", orig.shape, "-> после обработки:", img.shape)

    check_conv(gray, img)

    part_original(orig, img, gray)
    part_edges_all(gray)
    part_smoothing_all(img)
    part_sharpen_all(img, gray)
    part_sobel(gray)
    part_gauss(img)
    part_sharpen(img)


if __name__ == "__main__":
    main()
