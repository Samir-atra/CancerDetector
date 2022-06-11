import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import PIL.Image
import PIL
import pathlib
import matplotlib.pyplot as plt

data_path = pathlib.Path('/home/samer/Documents/Programming/AI50xIraq/Cancerdetection/archivecopy/Training/')
#data_dir = pathlib.Path(data_path)
data_path_val = pathlib.Path('/home/samer/Documents/Programming/AI50xIraq/Cancerdetection/archivecopy/Validation/')

dataset_path = tf.keras.utils.image_dataset_from_directory(
    data_path,
    labels= 'inferred',
    seed= 1,
    image_size=(180, 180),
    color_mode="grayscale",
    shuffle=True)

dataset_path_val = tf.keras.utils.image_dataset_from_directory(
    data_path_val,
    labels= 'inferred',
    seed= 2,
    image_size=(180, 180),
    color_mode="grayscale",
    shuffle=True)



#class_names = dataset_path.class_names
#print(class_names)

#data_path = pathlib.Path('/home/samer/Documents/Programming/AI50xIraq/Cancerdetection/archivecopy/Training')
#data_dir = pathlib.Path(data_path)
#image_count = len(list(data_path.glob('*/*.jpg')))
#print(image_count)

#plt.figure(figsize=(1, 2))
#for images, labels in dataset_path.take(1):
#    for i in range(2):
#        ax = plt.subplot(1, 2, i+1)
#        ax.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")
#    plt.show()
        #print("hello")
#x= 0
#for image_batch, labels_batch in dataset_path:
#    print(image_batch.shape)
#    print(labels_batch.shape)
#    x = x+1
#    continue
#print(x)

#image, label= next(iter(dataset_path.take(1)))
#_ = plt.imshow(image.numpy().astype("uint8"))
#_ = plt.title(get_label_name(label))

num_classes = 2

model = tf.keras.Sequential([
    #tf.keras.layers.Resizing(180, 180),
    tf.keras.layers.Rescaling(1./255),
    #tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    #tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.Conv2D(16, 3, activation='leaky_relu'),    
    tf.keras.layers.MaxPooling2D(),                        
    tf.keras.layers.Conv2D(32, 3, activation='leaky_relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='leaky_relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    #tf.keras.layers.Normalization()
])

#result = model(image)
#_ = plt.imshow(result)
#plt.show()

#plt.figure(figsize=(1, 1))
#for images, labels in dataset_path.take(1):
#    result = model(images[1])
#    print(result.numpy().min(), result.numpy().max())
#    _ = plt.imshow(result)
#    plt.show()


model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

model.fit(
    dataset_path,
    epochs=10,
    validation_data = dataset_path_val)

model.summary()







