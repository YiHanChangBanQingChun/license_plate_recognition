import imageio
import numpy as np
import os

# 1. 读取图片
def read_image(path):
    '''
    使用matplotlib的image模块或imageio来读取图像并转换为numpy数组。
    '''
    img = imageio.imread(path)
    return img

# 2. RGB转灰度
def rgb_to_grayscale(img):
    '''
    将RGB图像转换为灰度图像。
    '''
    if len(img.shape) == 3:
        grayscale = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        return grayscale.astype(np.uint8)
    else:
        return img

# 3. 高斯滤波
def gaussian_kernel(size, sigma=1):
    '''
    生成一个高斯滤波器。
    使用高斯滤波器或中值滤波器平滑图像，减少噪声干扰。
    '''
    
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i,j] = np.sum(region * kernel)
    return output

def denoise(image, kernel_size=3, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)

# 4. Sobel边缘检测
def sobel_filters(image):
    '''
    使用Sobel算子检测图像中的边缘，帮助定位车牌区域。
    '''
    Kx = np.array([[ -1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]])
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]])
    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G.astype(np.uint8), theta)

# 5. 阈值处理
def threshold(image, thresh=128):
    '''
    将灰度图转换为二值图，可以使用全局阈值或自适应阈值。
    '''
    binary = np.where(image > thresh, 255, 0).astype(np.uint8)
    return binary

def adaptive_threshold(image, block_size=15, C=10):
    # 简单的自适应阈值实现
    padded = np.pad(image, (block_size//2, block_size//2), mode='reflect')
    binary = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            block = padded[i:i+block_size, j:j+block_size]
            local_thresh = np.mean(block) - C
            binary[i,j] = 255 if image[i,j] > local_thresh else 0
    return binary

# 6. 膨胀和腐蚀
def dilate(image, kernel_size=3):
    '''
    使用膨胀和腐蚀操作增强车牌区域。
    '''
    kernel = np.ones((kernel_size, kernel_size))
    padded = np.pad(image, (kernel_size//2, kernel_size//2), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.any(region):
                output[i,j] = 255
    return output

def erode(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size))
    padded = np.pad(image, (kernel_size//2, kernel_size//2), mode='constant', constant_values=255)
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.all(region == 255):
                output[i,j] = 255
            else:
                output[i,j] = 0
    return output

# 7. 车牌识别
def find_plate_candidates(edges, binary):
    '''
    通过检测边缘和形态学操作，找到可能的车牌区域。
    通常车牌具有特定的长宽比和矩形特征。
    '''
    # 简单的连通区域检测
    from collections import deque

    visited = np.zeros_like(edges, dtype=bool)
    candidates = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i,j] == 255 and not visited[i,j]:
                queue = deque()
                queue.append((i,j))
                visited[i,j] = True
                region = []
                while queue:
                    x, y = queue.popleft()
                    region.append((x, y))
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < edges.shape[0] and 0 <= ny < edges.shape[1]:
                                if edges[nx, ny] == 255 and not visited[nx, ny]:
                                    queue.append((nx, ny))
                                    visited[nx, ny] = True
                if len(region) > 100:  # 过滤小区域
                    ys = [p[1] for p in region]
                    xs = [p[0] for p in region]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    aspect_ratio = (max_y - min_y) / (max_x - min_x + 1e-5)
                    if 2 < aspect_ratio < 6:  # 假设车牌的长宽比在2到6之间
                        candidates.append((min_x, min_y, max_x, max_y))
    return candidates

# 8. 车牌字符分割和识别
def crop_plate(image, bbox):
    '''
    从候选区域中裁剪出车牌，并进行必要的校正（如透视变换）。
    '''
    min_x, min_y, max_x, max_y = bbox
    return image[min_x:max_x, min_y:max_y]

# 9. 字符模板匹配
def segment_characters(plate):
    '''
    将车牌区域进行字符分割，可以通过垂直投影法实现。
    '''
    projection = np.sum(plate, axis=0)
    thresh = np.max(projection) * 0.5
    segments = []
    in_char = False
    start = 0
    for i, val in enumerate(projection):
        if val < thresh and not in_char:
            in_char = True
            start = i
        elif val >= thresh and in_char:
            in_char = False
            end = i
            if end - start > 5:  # 过滤过小区域
                segments.append((start, end))
    characters = [plate[:,s[0]:s[1]] for s in segments]
    return characters

# 10. 车牌识别
def load_char_templates(template_dir):
    '''
    采用模板匹配或特征匹配的方法进行字符识别。
    '''
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith('.npy'):
            char = filename.split('.')[0]
            templates[char] = np.load(os.path.join(template_dir, filename))
    return templates

def match_character(char_img, templates):
    char_img_resized = resize(char_img, (20, 20))  # 统一字符尺寸
    char_img_flat = char_img_resized.flatten()
    best_match = None
    min_diff = float('inf')
    for char, tmpl in templates.items():
        tmpl_flat = tmpl.flatten()
        diff = np.sum((char_img_flat - tmpl_flat) ** 2)
        if diff < min_diff:
            min_diff = diff
            best_match = char
    return best_match

def resize(image, size):
    # 简单的最近邻缩放
    resized = np.zeros(size, dtype=image.dtype)
    orig_h, orig_w = image.shape
    new_h, new_w = size
    for i in range(new_h):
        for j in range(new_w):
            orig_i = int(i * orig_h / new_h)
            orig_j = int(j * orig_w / new_w)
            resized[i,j] = image[orig_i, orig_j]
    return resized

# 11. 车牌识别
def recognize_plate(image_path, template_dir):
    '''
    将上述步骤整合，形成完整的车牌识别流程。
    '''
    img = read_image(image_path)
    gray = rgb_to_grayscale(img)
    denoised = denoise(gray)
    edges, _ = sobel_filters(denoised)
    binary = adaptive_threshold(denoised)
    dilated = dilate(binary)
    eroded = erode(dilated)
    candidates = find_plate_candidates(edges, eroded)
    templates = load_char_templates(template_dir)
    
    for bbox in candidates:
        plate_img = crop_plate(gray, bbox)
        plate_binary = adaptive_threshold(plate_img)
        characters = segment_characters(plate_binary)
        plate_text = ""
        for char_img in characters:
            recognized_char = match_character(char_img, templates)
            if recognized_char:
                plate_text += recognized_char
        if plate_text:
            return plate_text
    return None

# 12. 模板准备
def prepare_templates(source_dir, template_dir):
    '''
    为了进行模板匹配，需要准备每个字符的模板图像，并将其保存为numpy数组（例如，20x20像素）。
    这些模板可以通过手动收集和处理字符图像生成。
    '''
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            char = filename.split('.')[0]
            img = imageio.imread(os.path.join(source_dir, filename))
            gray = rgb_to_grayscale(img)
            binary = threshold(gray, thresh=128)
            resized = resize(binary, (20, 20))
            np.save(os.path.join(template_dir, f"{char}.npy"), resized)