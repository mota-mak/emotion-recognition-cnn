import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Задаем основные параметры
img_size = 48 # Размер изображения будет 48x48 пикселей
batch_size = 64
epochs = 15
num_classes = 7 # 7 классов эмоций

# Создаем генераторы данных для обучения и валидации
# Они автоматически масштабируют изображения (1/255)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        'data/train', # Целевая директория
        target_size=(img_size, img_size), # Приводим все изображения к одному размеру
        color_mode="grayscale", # Используем черно-белые изображения
        batch_size=batch_size,
        class_mode='categorical', # Категориальная классификация
        subset='training', # Указываем, что это тренировочная выборка
        shuffle=True)

validation_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation', # Указываем, что это валидационная выборка
        shuffle=True)

# Создаем генератор для тестовых данных (без аугментации и без разделения)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) # Не перемешиваем, чтобы потом сравнить с настоящими метками

# Создаем модель
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Компилируем модель
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Выводим summary модели
model.summary()

# Обучаем модель
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
    verbose=1     
)

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# Сохранение модели
model.save('emotion_recognition_model.h5')

# Построение графиков точности и потерь
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_history.png')
plt.show()
