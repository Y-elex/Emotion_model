import os
import numpy as np
from PIL import Image
import mediapipe as mp
from tqdm import tqdm

mp_face = mp.solutions.face_mesh
data = 'AffectNet'  # 可改为 FER-2013/AffectNet

root_dir = f"../datasets/{data}/test"  # 可改为 train/val/test
save_dir = f"./landmarks_{data}/test"           # 保存关键点

os.makedirs(save_dir, exist_ok=True)

for label in sorted(os.listdir(root_dir)):
    label_folder = os.path.join(root_dir, label)
    if not os.path.isdir(label_folder):
        continue
    save_label_folder = os.path.join(save_dir, label)
    os.makedirs(save_label_folder, exist_ok=True)

    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        for img_name in tqdm(os.listdir(label_folder), desc=f"Label {label}"):
            img_path = os.path.join(label_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            results = face_mesh.process(img_np)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                kp = np.array([[p.x, p.y] for p in landmarks.landmark], dtype=np.float32)
            else:
                kp = np.zeros((468, 2), dtype=np.float32)
            np.save(os.path.join(save_label_folder, img_name.replace(".jpg", ".npy")), kp)
