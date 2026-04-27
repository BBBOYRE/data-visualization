import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# 🌟 进阶参数字典：新增 min_circularity (圆度阈值)
# 圆度公式：4 * PI * Area / Perimeter^2
IMAGE_PARAMS = {
    "cell00.jpg": {
        "use_clahe": False, "method": "otsu", "thresh_val": 0, 
        "min_area": 15, "max_area": 5000, "min_circularity": 0.4, # 手绘图形状不规则，圆度给低一点
        "morph_op": "open", "morph_iter": 1
    },
    "cell01.jpg": {
        "use_clahe": False, "method": "fixed", "thresh_val": 94, 
        "min_area": 43, "max_area": 1693, "min_circularity": 0.7, # 细胞核很圆，设为0.7过滤掉长条细胞壁
        "morph_op": "open", "morph_iter": 3
    },
    "cell02.jpg": {
        "use_clahe": True, "method": "adaptive", "thresh_val": 0, 
        "min_area": 25, "max_area": 1000, "min_circularity": 0.6, # 红细胞较圆
        "morph_op": "close_open", "morph_iter": 1
    }
}

def calculate_circularity(contour):
    """计算轮廓圆度"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    # 圆度公式: 4 * PI * 面积 / 周长的平方
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity

def count_cells_pro(image_path, filename):
    if not os.path.exists(image_path): return None, None, 0

    # 读取与预处理 (支持中文路径)
    img_data = np.fromfile(image_path, dtype=np.uint8)
    original_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if original_img is None: return None, None, 0
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    params = IMAGE_PARAMS.get(filename, IMAGE_PARAMS["cell00.jpg"])
    
    if params["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_img = clahe.apply(gray_img)
        
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # 二值化
    if params["method"] == "otsu":
        _, thresh_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif params["method"] == "fixed":
        _, thresh_img = cv2.threshold(blurred_img, params["thresh_val"], 255, cv2.THRESH_BINARY_INV)
    else:
        thresh_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    
    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    if params["morph_op"] == "open":
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=params["morph_iter"])
    else:
        morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=params["morph_iter"])
        morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel, iterations=params["morph_iter"])
        
    # 轮廓检测
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = original_rgb.copy()
    cell_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        circularity = calculate_circularity(cnt)
        
        # 🌟 多维度判定：面积范围 + 圆度阈值
        if (params["min_area"] < area < params["max_area"]) and (circularity > params["min_circularity"]):
            cell_count += 1
            # 绘制绿色轮廓
            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
            
            # 标注序号与圆度值 (保留两位小数，便于报告分析)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(result_img, f"#{cell_count}({circularity:.2f})", (cX - 20, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    return original_rgb, result_img, cell_count

def main():
    base_path = "../期末报告图片素材/"
    image_files = ["cell00.jpg", "cell01.jpg", "cell02.jpg"]
    
    for filename in image_files:
        full_path = os.path.join(base_path, filename)
        print(f"\n[Pro模式] 正在处理: {filename} ...")
        
        original, result, count = count_cells_pro(full_path, filename)
        
        if original is not None:
            print(f"✅ {filename} 检测结果: {count} 个目标")
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1); plt.imshow(original); plt.title(f"Original"); plt.axis('off')
            plt.subplot(1, 2, 2); plt.imshow(result); plt.title(f"Pro Result: {count} Cells"); plt.axis('off')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()