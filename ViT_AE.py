import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras_vit import vit
from vit_utils import ViT_S8_AE

def load_images(path, num_images):
	x = []
	for i, file in enumerate(os.listdir(path)):
		image = cv2.imread(path + file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image / 255.0
		x.append(image)
		if i == num_images:
			break

	x = np.array(x)
	x = tf.cast(x, tf.float32)
	return x

def show_images(images, count):
	for i in range(count):
		img = (images[i]*255).astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def train_step(x):
	with tf.GradientTape() as tape:
		y_pred = model(x, training=True)

		loss = loss_func(x, y_pred)
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss

def val_step(x):
	y_pred = model(x, training=True)
	loss = loss_func(x, y_pred)
	return loss

def predict(x):
	return model(x, training=False)

image_path = '/home/marble/Desktop/Taylor/python/ml/product_imgs/bluebackground/train/'
val_path = '/home/marble/Desktop/Taylor/python/ml/product_imgs/bluebackground/val/'

show = False
train = False
IMG_SIZE = (224,224)
BATCH_SIZE = 8
LR = 0.0008
EPOCHS = 1000
steps_per_epoch =  10000 #1562 #6250 #12500
val_steps_per_epoch = 500
optimizer = keras.optimizers.Nadam(learning_rate=LR)
#loss_func = keras.losses.MeanAbsoluteError() # switched a epoch 3
loss_func = keras.losses.MeanSquaredError()
checkpoints = '/home/marble/Desktop/Taylor/python/ml/checkpoints/ViT_AE/v3/'
os.makedirs(checkpoints,exist_ok=True)

data_generator = ImageDataGenerator(rescale=1./255)

# Configure the data generator to load images from the directory
train_data_gen = data_generator.flow_from_directory(
	image_path,  # Directory where the images are located
	target_size=IMG_SIZE,  # Resize targets to ensure uniform image size
	color_mode='rgb',  # Load images in RGB mode
	batch_size=BATCH_SIZE,  # Number of images to load at each iteration
	class_mode=None,  # No labels are provided
	shuffle=True  # Shuffle the images 
)

val_data_gen = data_generator.flow_from_directory(
	val_path,  # Directory where the images are located
	target_size=IMG_SIZE,  # Resize targets to ensure uniform image size
	color_mode='rgb',  # Load images in RGB mode
	batch_size=BATCH_SIZE,  # Number of images to load at each iteration
	class_mode=None,  # No labels are provided
	shuffle=True  # Shuffle the images 
)

most_recent = 0
try:
	most_recent = max([int(model.split('_')[3].split('.keras')[0]) for model in os.listdir(checkpoints)])
	model = ViT_S8_AE()
	model.load_weights(f'{checkpoints}ViT_AE_v3_{most_recent}.keras')
	model.summary()
	print(f'Model found: ViT_AE_v3_{most_recent}.keras')
except Exception as e:
	print(e)
	#model = vit.ViT_S8_AE()
	model = ViT_S8_AE()
	model.summary()
	for layer in model.layers:
		if layer.name == 'conv_decoder':
			layer.summary()
			break

	print('No model found, creating new model')

model.compile(loss=loss_func, optimizer=optimizer)

if train:
	for epoch in range(EPOCHS):
		losses = []
		val_losses = []
		last_train_loss = 0
		last_train_step = 0
		print(f'Epoch {epoch+1}/{EPOCHS}')
		for step, x_batch in enumerate(train_data_gen):

			if x_batch.shape[0] < BATCH_SIZE:
				break
			
			if show:
				img2 = x_batch[1]
				cv2.imshow('image', cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			x_batch = tf.cast(x_batch, tf.float32)

			loss = train_step(x_batch)
			losses.append(loss)
			last_train_loss = np.mean(losses)
			last_train_step = step
			print(f'\rLoss: {last_train_loss:.4f}, {step}/{steps_per_epoch}', end='', flush=True)

			if step == steps_per_epoch:
				break

		for step, val_batch in enumerate(val_data_gen):
			
			if val_batch.shape[0] < BATCH_SIZE:
				break
			
			if show:
				img2 = x_batch[1]
				cv2.imshow('image', cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			val_batch = tf.cast(val_batch, tf.float32)

			loss = val_step(x_batch)
			val_losses.append(loss)
			print(f'\rLoss: {last_train_loss:.4f}, {last_train_step}/{steps_per_epoch}, Val Loss: {np.mean(val_losses):.4f}, {step}/{val_steps_per_epoch}', end='', flush=True)

			if step == val_steps_per_epoch:
				break
			
		print()  # Ensure the epoch summary is printed on a new line

		model.save_weights(f'{checkpoints}ViT_AE_v2_{most_recent+epoch+1}.keras')
else:
	x = val_data_gen[0]  
	y = predict(x)

	# Plotting
	fig, axs = plt.subplots(2, 4, figsize=(20, 10))
	
	for i in range(4):
		if i < len(x) and i < len(y):
			# Top row: Original images from x
			axs[0, i].imshow(x[i])
			axs[0, i].axis('off')
	
			# Bottom row: Predictions from y
			axs[1, i].imshow(y[i])
			axs[1, i].axis('off')
	
	plt.tight_layout()
	plt.show()
