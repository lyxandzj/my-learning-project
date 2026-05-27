import warnings

from ultralytics import YOLO

# 忽略一些不必要的警告信息，保持控制台整洁
warnings.filterwarnings("ignore")


def main():
    # 1. 加载模型
    # 前期跑基线：直接使用官方预训练权重 yolo11n.pt
    # 后期跑创新点：使用我们自定义的网络结构 yaml 文件，例如 YOLO('yolo11-UAV.yaml')
    # model = YOLO('yolo11n.pt')
    model = YOLO("yolo11-BiFPN-Ghost.yaml").load("yolo11n.pt")

    # 2. 开始训练
    print("🚀 开始在 RTX 4090 上训练无人机目标检测模型...")
    model.train(
        # ================= 数据与设备配置 =================
        data="VisDrone.yaml",  # 你的数据集配置文件路径
        device=0,  # 指定第一块 GPU（4090）
        workers=8,  # 数据加载线程数，4090算力强，建议设为 8 或 16 以防 CPU 拖后腿
        # ================= 核心训练参数 =================
        epochs=150,  # 总训练轮数，150~200 足够了
        patience=30,  # 早停机制：如果连续 30 个 epoch mAP 没有提升，自动停止训练，节省时间
        # 【极其重要】针对 4090 和 小目标的联合优化
        imgsz=1024,  # 航拍图像极速提点的秘诀：原图分辨率通常很大，将默认的 640 提升至 1024，能极大减少小目标像素丢失！
        batch=8,  # 4090 有 24G 显存，跑 1024 分辨率的 Nano 模型，Batch 设为 32 或 16 毫无压力，梯度更稳定。
        cache=False,
        # ================= 优化器与保存机制 =================
        optimizer="auto",  # 自动选择优化器（通常是 AdamW 或 SGD）
        save=True,  # 默认开启，自动在 runs/detect/train/weights 下保存 best.pt 和 last.pt
        save_period=-1,  # 不保存中间 epoch 的权重，只留 best 和 last，节省硬盘空间
        project="UAV_Detection",  # 实验结果保存的根目录名称
        name="yolo11-BiFPN-Ghost",  # 当前实验的名称，下次跑改进模型可以改为 'yolo11_pconv_ema'
        exist_ok=True,  # 允许覆盖同名文件夹
        # ================= 数据增强（针对小目标定制） =================
        mosaic=1.0,  # 开启马赛克数据增强，把4张图拼成1张，非常利于模型学习小目标
        close_mosaic=10,  # YOLO 核心 Trick：在最后 10 个 epoch 关闭马赛克，让模型回归真实图片分布，能再提一点精度
        mixup=0.1,  # 引入少量 MixUp 增强，提升泛化能力
    )

    print("✅ 训练完成！最优模型权重已自动保存。")


if __name__ == "__main__":
    # Windows/Linux 系统下多线程 DataLoader 必须放在 __main__ 保护块中
    main()
