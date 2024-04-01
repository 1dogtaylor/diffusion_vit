import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras_vit import vit

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

def show_noisy_images(images, count):
	for i in range(count):
		img_a = (images[i][0]*255).astype(np.uint8)
		img_b = (images[i][1]*255).astype(np.uint8)
		img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
		img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR)
		cv2.imshow('image at t', img_a)
		cv2.imshow('image at t+1', img_b)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def forward_noise(x, timestamp):
	a = time_bar[timestamp]
	b = time_bar[timestamp+1]
	noise = np.random.normal(scale=1.0, size=x.shape)  # noise mask
	a = a.reshape((-1, 1, 1, 1))
	b = b.reshape((-1, 1, 1, 1))
	# Directly add scaled noise to the images
	img_a = x * (1 - a) + noise * a
	img_b = x * (1 - b) + noise * b
	return img_a, img_b

def forward_noise_2(x, timestamp):
	a = time_bar[timestamp]
	noise = np.random.normal(loc=0.5, size=x.shape)  # noise mask
	noise = np.clip(noise, 0, 1)
	a = a.reshape((-1, 1, 1, 1))
	img_a = x * (1 - a) + noise * a
	return img_a, noise

def train_step(x, x_ts, y):
	with tf.GradientTape() as tape:
		y_pred = model([x, tf.expand_dims(x_ts,axis=-1)], training=True)
		loss = loss_func(y, y_pred)
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss

def predict(x, x_ts):
	return model([x, x_ts], training=False)

image_path = '/home/marble/Desktop/Taylor/python/ml/diffusion/celeba/'

show = False
train = True
IMG_SIZE = (224,224)
BATCH_SIZE = 32
LR = 0.0008
EPOCHS = 100
steps_per_epoch =  3125 #1562 #6250 #12500
checkpoints = '/home/marble/Desktop/Taylor/python/ml/diffusion/checkpoints/'
timesteps = 256    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps

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


try:
	most_recent = max([int(model.split('diffusion_vit_v1_')[1].split('.keras')[0]) for model in os.listdir(checkpoints)])
	model = vit.ViT_B16_Diffusion()
	model.load_weights(f'{checkpoints}diffusion_vit_v1_{most_recent}.keras')
	model.summary()
	print(f'Model found: diffusion_vit_v1_{most_recent}.h5')
except Exception as e:
	print(e)
	model = vit.ViT_B16_Diffusion()
	model.summary()
	print('No model found, creating new model')



optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
loss_func = tf.keras.losses.MeanSquaredError()
model.compile(loss=loss_func, optimizer=optimizer)
if train:
	for epoch in range(EPOCHS):
		losses = []
		print(f'Epoch {epoch+1}/{EPOCHS}')
		for step, x_batch in enumerate(train_data_gen):

			if x_batch.shape[0] < BATCH_SIZE:
				break
			
			if show:
				img1 = x_batch[7]
				img2 = x_batch[1]
				cv2.imshow('image1', cv2.cvtColor((img1*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imshow('image2', cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

			x_ts_batch = np.random.randint(0, timesteps, BATCH_SIZE)
			#x_ts_batch = np.array([255,245,230,220,200,180,160,1])

			x_batch, y_batch = forward_noise_2(x_batch, x_ts_batch)
			

			if show:
				print(np.max(x_batch),np.min(x_batch))
				print(np.max(y_batch), np.min(y_batch))
				cv2.imshow('x1', cv2.cvtColor((x_batch[5]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imshow('y1', cv2.cvtColor((y_batch[5]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imshow('x2', cv2.cvtColor((x_batch[1]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imshow('y2', cv2.cvtColor((y_batch[1]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

				diff = x_batch - 0.01*y_batch

				cv2.imshow('diff1', cv2.cvtColor((diff[5]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imshow('diff2', cv2.cvtColor((diff[1]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

				cv2.waitKey(0)
				cv2.destroyAllWindows()

			x_batch = tf.cast(x_batch, tf.float32)
			x_ts_batch = tf.cast(x_ts_batch, tf.float32)
			y_batch = tf.cast(y_batch, tf.float32)

			loss = train_step(x_batch, x_ts_batch ,y_batch)
			losses.append(loss)
			print(f'\rLoss: {np.mean(losses):.4f}, {step}/{steps_per_epoch}', end='', flush=True)

			if step == steps_per_epoch:
				break
			
		print()  # Ensure the epoch summary is printed on a new line
		print(f'Epoch {epoch+1}, Average Loss: {np.mean(losses):.4f}')

		model.save_weights(f'{checkpoints}diffusion_vit_v1_{epoch+1}.keras')
else:
	x = np.random.randn(1, IMG_SIZE[0], IMG_SIZE[1], 3)  # Assuming 3 channels for RGB

	# Generate 256 steps
	images = [x]
	print('Generating image...')
	for step in range(255):
		# Simulate the reverse diffusion process
		# Generate a time step value. This part depends on how your model expects the time step to be formatted.
		# Here we assume it's normalized between 0 and 1.
		x_ts = np.array([[step / 255.0]])
		x = predict(x, x_ts)
		images.append(x)
		print(f'\rstep: {step+1}/256', end='', flush=True)
	
	# Plotting
	fig, axs = plt.subplots(16, 16, figsize=(20, 20))
	for i, ax in enumerate(axs.flat):
		# Assuming the image is returned in the right shape and scale from predict
		# You might need to rescale or adjust the image format depending on your model's output
		ax.imshow(images[i][0])
		ax.axis('off')
	plt.tight_layout()
	plt.show()

#----------------------------------- img out
# Epoch 1, Average Loss: 0.3431
# Epoch 2, Average Loss: 0.3383
# Epoch 3, Average Loss: 0.3384
# Epoch 4, Average Loss: 0.3378
# Epoch 1, Average Loss: 0.3372
# Epoch 2, Average Loss: 0.3365
# Epoch 4, Average Loss: 0.3371
# Epoch 1, Average Loss: 0.3360
# Epoch 2, Average Loss: 0.3351
# Epoch 1, Average Loss: 0.3356
# Epoch 1, Average Loss: 0.3344
# Epoch 2, Average Loss: 0.3363
# Epoch 3, Average Loss: 0.3351
# Epoch 4, Average Loss: 0.3358
	
#----------------------------------- noise out
	# mae
# Epoch 1, Average Loss: 0.6930
# Epoch 1, Average Loss: 0.6848
# Epoch 2, Average Loss: 0.6835
# Epoch 4, Average Loss: 0.6828
	
	# mse
# Epoch 1, Average Loss: 0.7709
# Epoch 2, Average Loss: 0.7692
# Epoch 3, Average Loss: 0.7697
# Epoch 4, Average Loss: 0.7688
# Epoch 5, Average Loss: 0.7683
# Epoch 7, Average Loss: 0.7682


# Loss: 0.1976, 1566/3125