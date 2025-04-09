import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. CIFAR-10 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 전처리: 정규화 + one-hot 인코딩
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. CNN 모델 구성
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout((0.25)),
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout((0.25)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 학습 (에폭 50)
history = model.fit(x_train, y_train, epochs=50, batch_size=64,
                    validation_split=0.1)

# 6. 테스트 정확도
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n[✓] 테스트 정확도: {test_acc:.4f}")

# 7. 정확도 그래프 출력
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CIFAR-10 CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 8. testimg.jpg 로 예측하기
def predict_custom_image(img_path):
    img = Image.open(img_path).resize((32, 32))
    img_array = np.array(img) / 255.0  # 정규화
    if img_array.shape != (32, 32, 3):
        raise ValueError("이미지는 32x32 크기의 컬러(RGB) 이미지여야 합니다.")
    
    img_input = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    prediction = model.predict(img_input)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"[✓] 예측 결과: {class_names[pred_class]} ({confidence:.2%} 확신)")

# 9. 실제 이미지 예측
predict_custom_image("./data/testimg.jpg")
