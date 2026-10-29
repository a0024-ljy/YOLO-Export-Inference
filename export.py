from ultralytics import YOLO
import time

# ===== 第 1 步：导出 onnx =====
model = YOLO("best.pt")        # ← 如果你的权重文件名不是 best.pt，改成实际名字
model.export(format="onnx", imgsz=640)

# ===== 第 2 步：两种模型各测速 20 次 =====
for w in ["best.pt", "best.onnx"]:
    m = YOLO(w)
    m.predict("frame_00.jpg", verbose=False)   # 预热一次
    t0 = time.time()
    for _ in range(20):
        m.predict("frame_00.jpg", verbose=False)
    avg = (time.time() - t0) / 20 * 1000
    print(f"{w}: 平均 {avg:.1f} 毫秒/张")

