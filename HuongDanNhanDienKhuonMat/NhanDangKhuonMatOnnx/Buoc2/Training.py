import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# Lấy đường dẫn tuyệt đối tới thư mục hiện tại
script_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tuyệt đối đến các file model
fd_path = os.path.abspath(os.path.join(script_dir, '..', 'model', 'face_detection_yunet_2023mar.onnx'))
fr_path = os.path.abspath(os.path.join(script_dir, '..', 'model', 'face_recognition_sface_2021dec.onnx'))

class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            ext = os.path.splitext(f)[1]
            if ext.lower() in ['.jpg', '.jpeg', '.bmp']:
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    return img[..., ::-1]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    plt.show()

# Load detector và recognizer từ đường dẫn tuyệt đối
detector = cv2.FaceDetectorYN.create(fd_path, "", (320, 320), 0.9, 0.3, 5000)
detector.setInputSize((320, 320))
recognizer = cv2.FaceRecognizerSF.create(fr_path, "")

# Load dữ liệu ảnh từ thư mục image
image_dir = os.path.abspath(os.path.join(script_dir, '..', 'image'))
metadata = load_metadata(image_dir)
embedded = np.zeros((metadata.shape[0], 128))

# Trích xuất đặc trưng khuôn mặt từ từng ảnh
for i, m in enumerate(metadata):
    print(f"Processing: {m.image_path()}")
    img = cv2.imread(m.image_path(), cv2.IMREAD_COLOR)
    face_feature = recognizer.feature(img)
    embedded[i] = face_feature

# Mã hóa nhãn và chia tập train/test
targets = np.array([m.name for m in metadata])
encoder = LabelEncoder()
encoder.fit(targets)
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 5 != 0
test_idx = np.arange(metadata.shape[0]) % 5 == 0
X_train = embedded[train_idx]
X_test = embedded[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# Huấn luyện SVM và lưu model
svc = LinearSVC()
svc.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc.predict(X_test))
print('SVM accuracy: %.6f' % acc_svc)

# Lưu model và encoder
model_path = os.path.abspath(os.path.join(script_dir, '..', 'model', 'svc.pkl'))
joblib.dump((svc, encoder), model_path)
