#!/usr/bin/env python3
"""
将 output/demo_4/imgs 文件夹中的图片合成为视频
"""
import cv2
import os
import glob
from tqdm import tqdm

def create_video(img_dir='output/demo_4/imgs', output_file='output/demo_4/demo_4_video.mp4', fps=30):
    """
    将指定目录中的图片合成为视频
    
    参数:
        img_dir: 图片所在目录
        output_file: 输出视频文件路径
        fps: 视频帧率（每秒帧数）
    """
    print(f"正在检查图片目录: {img_dir}...")
    
    # 1. 收集所有 PNG 图片并排序
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not images:
        print(f"❌ 错误: 在 {img_dir} 中未找到图片!")
        return
    
    print(f"✅ 找到 {len(images)} 张图片，开始合成视频...")
    
    # 2. 读取第一张图片获取尺寸
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"❌ 错误: 无法读取图片 {images[0]}")
        return
    
    height, width, layers = frame.shape
    size = (width, height)
    
    print(f"图片尺寸: {width} x {height}")
    
    # 3. 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 4. 创建视频写入器
    # 使用 mp4v 编码器，兼容性较好
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    if not out.isOpened():
        print("❌ 错误: 无法创建视频写入器")
        return
    
    # 5. 逐帧写入视频
    for filename in tqdm(images, desc="合成视频"):
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"⚠️  警告: 跳过无法读取的图片 {filename}")
    
    # 6. 释放资源
    out.release()
    
    print(f"\n🎉 成功! 视频已保存到: {output_file}")
    print(f"视频信息:")
    print(f"  - 总帧数: {len(images)}")
    print(f"  - 帧率: {fps} fps")
    print(f"  - 时长: {len(images)/fps:.2f} 秒")
    print(f"  - 分辨率: {width} x {height}")

if __name__ == "__main__":
    create_video()
