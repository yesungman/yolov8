yolov8로 1.jpg 인식하기
# 1. YOLOv8 설치
!pip install -q ultralytics

# 2. yolo8n.pt 파일 다운로드
!wget -O /content/yolo8n.pt https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt

# 3. 이미지 업로드 (1.jpg를 직접 업로드할 수 있게 함)
from google.colab import files
uploaded = files.upload()  # 너의 바탕화면에 있는 1.jpg를 업로드해

# 4. YOLO 모델 불러오기
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# 모델 로드
model = YOLO('/content/yolov8n.pt')

# 파일명이 '1.jpg'라고 가정
image_path = '/content/1.jpg'

# 5. 이미지 인식
results = model(image_path)

# 6. 결과 이미지 저장
annotated_img = results[0].plot()
output_path = '/content/output.jpg'
Image.fromarray(annotated_img).save(output_path)

# 7. 결과 이미지 출력
img = Image.open(output_path)
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis('off')
plt.title('YOLOv8 Detection Result')
plt.show()
