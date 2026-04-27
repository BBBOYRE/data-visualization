"""
题目2：确定图片中圆形物体的圆心位置
核心算法：霍夫圆检测 → 圆心坐标提取 → 标注
依赖库：OpenCV, NumPy, Matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ── 全局设置 ──────────────────────────────────────────────
matplotlib.rcParams['font.sans-serif'] = ['SimHei']      # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False         # 负号显示
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 每张图片的霍夫圆检测参数（已调优）──────────────────────
# 说明：
#   dp       - 累加器分辨率与图像分辨率之比（1 表示相同分辨率，值越大越粗略）
#   minDist  - 检测到的圆心之间的最小距离（像素）
#   param1   - Canny 边缘检测的高阈值
#   param2   - 累加器投票阈值，越大越严格（减少误检）
#   minRadius, maxRadius - 圆的半径搜索范围（像素）
#   blur     - 预处理滤波方法 ("gaussian" 或 "median")
PARAMS = {
    # fig03: 排球（大圆，表面有缝线纹理→需要重高斯模糊 + 跳过CLAHE避免重新增强纹理）
    "fig03.png": dict(dp=2, minDist=300, param1=100, param2=200,
                      minRadius=300, maxRadius=700,
                      blur="gaussian", blur_ksize=15, blur_sigma=3,
                      skip_clahe=True),
    # fig04: 硬币（小圆，边缘清晰→CLAHE 增强后中值滤波即可）
    "fig04.jpg": dict(dp=1.2, minDist=50,  param1=100, param2=45,
                      minRadius=50,  maxRadius=180,
                      blur="median", blur_ksize=5),
}
# 未在上表中配置的图片使用以下默认参数
DEFAULT_PARAMS = dict(dp=1.2, minDist=80, param1=100, param2=40,
                      minRadius=10, maxRadius=300,
                      blur="median", blur_ksize=5)


# ═══════════════════════════════════════════════════════════
#  1. 读取图像（支持中文路径）
# ═══════════════════════════════════════════════════════════
def read_image(image_path):
    """使用 np.fromfile + imdecode 安全读取中文路径图片"""
    if not os.path.exists(image_path):
        print(f"❌ 找不到图片: {image_path}")
        return None
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ 无法解码图片: {image_path}")
    return img


# ═══════════════════════════════════════════════════════════
#  2. 图像预处理
# ═══════════════════════════════════════════════════════════
def preprocess(img, params):
    """
    预处理流程：
      (1) BGR → 灰度
      (2) 滤波去噪（根据 params 选择高斯 / 中值滤波）
      (3) CLAHE 自适应直方图均衡化（增强对比度，改善低对比度圆形边缘检测）
    返回：gray_raw（灰度原图）、gray_blur（滤波后）、gray_eq（均衡化后）
    """
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 根据配置选择滤波方式
    blur_method = params.get('blur', 'median')
    ksize = params.get('blur_ksize', 5)
    if blur_method == 'gaussian':
        sigma = params.get('blur_sigma', 2)
        gray_blur = cv2.GaussianBlur(gray_raw, (ksize, ksize), sigma)
    else:
        gray_blur = cv2.medianBlur(gray_raw, ksize)

    # CLAHE 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    return gray_raw, gray_blur, gray_eq


# ═══════════════════════════════════════════════════════════
#  3. 霍夫圆检测
# ═══════════════════════════════════════════════════════════
def detect_circles(gray, params):
    """
    调用 cv2.HoughCircles 进行圆形检测。
    返回：circles 数组（每行 [x, y, r]）或 None
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=params['dp'],
        minDist=params['minDist'],
        param1=params['param1'],
        param2=params['param2'],
        minRadius=params['minRadius'],
        maxRadius=params['maxRadius'],
    )
    return circles


# ═══════════════════════════════════════════════════════════
#  4. 在图像上标注检测结果
# ═══════════════════════════════════════════════════════════
def annotate(img, circles):
    """
    在原图上绘制：
      - 圆的轮廓（绿色）
      - 圆心点  （红色）
      - 圆心坐标文字（蓝色）
      - 十字准线（黄色，增强可视性）
    返回标注后的图像副本和坐标列表。
    """
    output = img.copy()
    coords = []

    if circles is None:
        return output, coords

    circles_rounded = np.uint16(np.around(circles))
    for c in circles_rounded[0, :]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        coords.append((cx, cy, r))

        # 根据图像大小自适应线宽和字体
        h, w = img.shape[:2]
        scale = max(h, w) / 800          # 基准 800px
        thickness_circle = max(2, int(3 * scale))
        thickness_cross  = max(1, int(2 * scale))
        font_scale        = max(0.5, 0.7 * scale)
        cross_len         = max(8, int(15 * scale))
        center_radius     = max(3, int(5 * scale))

        # 绘制圆轮廓（绿色）
        cv2.circle(output, (cx, cy), r, (0, 255, 0), thickness_circle)

        # 绘制圆心实心点（红色）
        cv2.circle(output, (cx, cy), center_radius, (0, 0, 255), -1)

        # 绘制十字准线（黄色）
        cv2.line(output, (cx - cross_len, cy), (cx + cross_len, cy),
                 (0, 255, 255), thickness_cross)
        cv2.line(output, (cx, cy - cross_len), (cx, cy + cross_len),
                 (0, 255, 255), thickness_cross)

        # 标注坐标文字（蓝色，带白色背景提高可读性）
        label = f"({cx}, {cy})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, max(1, int(scale)))
        tx = cx - tw // 2
        ty = cy - int(20 * scale)
        # 白色背景矩形
        cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                      (255, 255, 255), -1)
        cv2.rectangle(output, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4),
                      (0, 0, 0), max(1, int(scale)))
        # 坐标文字
        cv2.putText(output, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
                    max(1, int(2 * scale)), cv2.LINE_AA)

    return output, coords


# ═══════════════════════════════════════════════════════════
#  5. 可视化 — 预处理流程图（4 步）
# ═══════════════════════════════════════════════════════════
def plot_preprocessing(img_rgb, gray_raw, gray_blur, gray_eq, edges, filename, params):
    """展示完整的预处理管线"""
    blur_name = "高斯滤波" if params.get('blur') == 'gaussian' else "中值滤波"
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    titles = ['① 原始图像', '② 灰度化', f'③ {blur_name}', '④ CLAHE 均衡化', '⑤ Canny 边缘']
    images = [img_rgb, gray_raw, gray_blur, gray_eq, edges]
    cmaps  = [None, 'gray', 'gray', 'gray', 'gray']

    for ax, im, t, cm in zip(axes, images, titles, cmaps):
        ax.imshow(im, cmap=cm)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.axis('off')

    fig.suptitle(f'图像预处理流程 — {filename}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"preprocess_{os.path.splitext(filename)[0]}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  💾 预处理流程图已保存: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  6. 可视化 — 原图 vs 检测结果对比图
# ═══════════════════════════════════════════════════════════
def plot_result(img_rgb, result_rgb, coords, filename):
    """左右对比：原图 + 检测标注图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.imshow(img_rgb)
    ax1.set_title('原始图像', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(result_rgb)
    n = len(coords)
    ax2.set_title(f'圆心检测结果（共 {n} 个圆）', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 在图下方输出坐标表格
    coord_text = "  |  ".join([f"圆{i+1}: ({c[0]}, {c[1]})  r={c[2]}" for i, c in enumerate(coords)])
    fig.text(0.5, 0.01, coord_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

    fig.suptitle(f'题目 2 · 圆心定位 — {filename}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, f"result_{os.path.splitext(filename)[0]}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  💾 检测结果图已保存: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════
def main():
    base_path = os.path.join(os.path.dirname(__file__), "..", "期末报告图片素材")
    image_files = ["fig03.png", "fig04.jpg"]

    for filename in image_files:
        full_path = os.path.join(base_path, filename)
        print(f"\n{'='*60}")
        print(f"  正在处理: {filename}")
        print(f"{'='*60}")

        # ── Step 1: 读取 ──
        img = read_image(full_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        print(f"  📐 图像尺寸: {w} × {h}")

        # ── Step 2: 预处理 ──
        params = PARAMS.get(filename, DEFAULT_PARAMS)
        gray_raw, gray_blur, gray_eq = preprocess(img, params)
        # 选择用于检测的灰度图（跳过 CLAHE 时使用滤波后图像）
        gray_for_detect = gray_blur if params.get('skip_clahe') else gray_eq
        # 生成 Canny 边缘图（仅供可视化展示，霍夫检测内部自行调用）
        edges = cv2.Canny(gray_for_detect, params['param1'] // 2, params['param1'])
        blur_name = "高斯滤波" if params.get('blur') == 'gaussian' else "中值滤波"
        clahe_note = "（跳过 CLAHE）" if params.get('skip_clahe') else " → CLAHE 均衡化"
        print(f"  ✅ 预处理完成（灰度化 → {blur_name}{clahe_note}）")

        # ── Step 3: 霍夫圆检测 ──
        circles = detect_circles(gray_for_detect, params)
        n_detected = 0 if circles is None else circles.shape[1]
        print(f"  🔍 霍夫圆检测参数: {params}")
        print(f"  🎯 检测到 {n_detected} 个圆")

        # ── Step 4: 标注 ──
        annotated, coords = annotate(img, circles)
        for i, (cx, cy, r) in enumerate(coords):
            print(f"     圆 {i+1}: 圆心 ({cx}, {cy}), 半径 {r} px")

        # ── Step 5: 可视化 ──
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        plot_preprocessing(img_rgb, gray_raw, gray_blur, gray_eq, edges, filename, params)
        plot_result(img_rgb, result_rgb, coords, filename)

    print(f"\n{'='*60}")
    print(f"  ✅ 全部处理完成！结果图片保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()