import cv2
import os
from PIL import Image

# Sử dụng camera gắn với thiết bị có ID là 0
cap = cv2.VideoCapture(0)

# Thư mục lưu trữ các frame
output_directory = r'C:\Users\lecon\OneDrive\Máy tính\Face Regconition\archive\Celebrity Faces Dataset\Nhat Anh'

# Tạo thư mục nếu nó chưa tồn tại
os.makedirs(output_directory, exist_ok=True)

frame_count = 0
max_frames = 100  # Số lượng frame tối đa trước khi tắt camera

while frame_count < max_frames:
    ret, frame = cap.read()

    if not ret:
        break
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_count += 1

    # Lưu frame vào thư mục
    frame_filename = f"{output_directory}/frame_{frame_count}.jpg"
    pil_image.save(frame_filename)

    # Hiển thị frame trên màn hình
    cv2.imshow('Frame', frame)

    # Thoát vòng lặp nếu người dùng nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
