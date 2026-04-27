import cv2
import numpy as np
import os

def nothing(x):
    pass

def run_tuner(image_filename):
    # 构建路径并读取图片
    image_path = f"../期末报告图片素材/{image_filename}"
    if not os.path.exists(image_path):
        print(f"找不到图片: {image_path}")
        return

    img_data = np.fromfile(image_path, dtype=np.uint8)
    original_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 高斯去噪 (固定大小)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # 创建调参窗口
    cv2.namedWindow('Tuning Dashboard', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tuning Dashboard', 600, 250)

    # 创建滑动条 (初始值, 最大值, 回调函数)
    cv2.createTrackbar('Threshold', 'Tuning Dashboard', 120, 255, nothing)
    cv2.createTrackbar('Min Area', 'Tuning Dashboard', 15, 500, nothing)
    cv2.createTrackbar('Max Area', 'Tuning Dashboard', 1000, 5000, nothing)
    cv2.createTrackbar('Morph Iter', 'Tuning Dashboard', 1, 5, nothing)

    print(f"正在调优: {image_filename}")
    print("操作说明：拖动滑动条实时查看效果。按 'ESC' 键退出。")

    while True:
        # 获取当前滑动条的值
        thresh_val = cv2.getTrackbarPos('Threshold', 'Tuning Dashboard')
        min_area = cv2.getTrackbarPos('Min Area', 'Tuning Dashboard')
        max_area = cv2.getTrackbarPos('Max Area', 'Tuning Dashboard')
        morph_iter = cv2.getTrackbarPos('Morph Iter', 'Tuning Dashboard')

        # 1. 应用固定阈值二值化
        ret, thresh = cv2.threshold(blurred_img, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # 2. 形态学开运算 (去噪点和细线)
        kernel = np.ones((3, 3), np.uint8)
        if morph_iter > 0:
            morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        else:
            morph = thresh

        # 3. 轮廓检测
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. 绘制结果
        result_img = original_img.copy()
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                count += 1
                cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)

        # 在左上角显示当前计数
        cv2.putText(result_img, f"Cells: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 将二值化黑白图转为彩色通道，以便和结果图拼接在一起对比
        morph_color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        
        # 左右拼接图像 (左边是黑白掩膜，右边是识别结果)
        combined = np.hstack((morph_color, result_img))
        
        # 缩小一半显示，防止屏幕放不下
        display_img = cv2.resize(combined, (0, 0), fx=1, fy=1)

        cv2.imshow('Live Preview (Left: Mask, Right: Result)', display_img)

        # 检测键盘输入，按 ESC 退出
        if cv2.waitKey(1) & 0xFF == 27:
            print(f"最终参数 -> Threshold: {thresh_val}, Min Area: {min_area}, Max Area: {max_area}, Morph Iter: {morph_iter}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 在这里更改你要调优的图片名称！
    run_tuner("cell01.jpg")