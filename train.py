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
