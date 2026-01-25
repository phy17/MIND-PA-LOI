import cv2
import os
import glob
from tqdm import tqdm

def make_video(img_dir='outputs/demo_1/imgs', output_file='outputs/demo_1/video_rescued.mp4', fps=30):
    print(f"Checking images in {img_dir}...")
    
    # 1. 收集图片
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not images:
        print("❌ Error: No images found!")
        return

    print(f"✅ Found {len(images)} images. Stitching video...")

    # 2. 读取第一张图获取尺寸
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    # 3. 创建写入器
    # 使用 mp4v 编码，兼容性好
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # 4. 写入帧
    for filename in tqdm(images):
        img = cv2.imread(filename)
        out.write(img)

    out.release()
    print(f"🎉 Success! Video saved to: {output_file}")
    
    # 尝试把用户原来的损坏状态标记为成功（可选）
    print("Suggest opening the folder to view result.")

if __name__ == "__main__":
    make_video()
