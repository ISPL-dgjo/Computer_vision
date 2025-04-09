import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# 1. CIFAR-10 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 사전학습된 VGG16 로드 (include_top=False -> 최상위 FC 층 제거)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 3. 모든 레이어를 freeze
for layer in base_model.layers:
    layer.trainable = False

# 4. 전이학습 모델 구성
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 5. 컴파일 및 학습
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=30,
                    batch_size=64)

# 6. 테스트 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n[✓] 테스트 정확도: {test_acc:.4f}")

# 7. 정확도 그래프 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('VGG16 전이학습 모델 정확도')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()