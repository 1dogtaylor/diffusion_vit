""""
This implementation of "a picture is worth 16x16 words" is a modified version of https://github.com/djsaber/Keras-ViT/tree/main which is under MIT license
"""

from keras import layers
from keras import activations
from keras import backend as K
from keras import utils
from keras import models, Model
from keras import layers
from keras import activations
import tensorflow as tf

class PatchEmbedding(layers.Layer):
    """patch embedding layer"""
    def __init__(self, patch_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.hidden_dim, self.patch_size, self.patch_size)
        self.flatten = layers.Reshape((-1, self.hidden_dim))
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return x

class AddCLSToken(layers.Layer):
    """add class token layer"""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token_weight",
            shape=(1, 1, self.hidden_dim),
            initializer="zero",
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):   
        x = K.tile(self.class_token, [K.shape(inputs)[0],1,1])
        x = K.concatenate([x, inputs], axis=1)
        return x
    
class AddMultiCLSToken(layers.Layer):
    """Add multiple class token layer"""
    def __init__(self, hidden_dim, num_tokens=6, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens  # Number of class tokens to add
    def build(self, input_shape):
        # Adjust the shape to include multiple tokens
        self.class_token = self.add_weight(
            name="class_tokens_weight",
            shape=(1, self.num_tokens, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)
    def call(self, inputs):
        # Tile the class token to match the batch size
        class_tokens = K.tile(self.class_token, [K.shape(inputs)[0], 1, 1])
        # Concatenate the class tokens with the inputs
        return K.concatenate([class_tokens, inputs], axis=1)

class AddCLSToken_diffusion(layers.Layer):
    """add class token layer"""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token_weight",
            shape=(1, 1, self.hidden_dim),
            initializer="zero",
            trainable=True
            )
        
        # Timestamp projection weights
        self.time_projection = self.add_weight(
            name="time_projection_weight",
            shape=(1, self.hidden_dim),  # Projecting 1D timestamp to hidden_dim
            initializer="zeros",
            trainable=True,
        )
        return super().build(input_shape)

    def call(self, inputs):
        inputs, ts = inputs

        # Project the timestamp to the hidden dimension
        projected_time = K.dot(K.expand_dims(ts, -1), self.time_projection)
        
        cls = K.tile(self.class_token, [K.shape(inputs)[0],1,1])
        
        x = K.concatenate([projected_time, cls, inputs], axis=1)
        return x

class AddPositionEmbedding(layers.Layer):
    """add position encoding layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.pe = self.add_weight(
            name="position_embedding",
            shape=(1, input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pe

class MultiHeadAttention(layers.Layer):
    """multi head self attention layer"""
    def __init__(
        self, 
        heads,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.heads = heads

    def build(self, input_shape):
        self.dk = K.sqrt(K.cast(input_shape[-1]//self.heads, dtype=tf.float32))
        self.q_dense = layers.Dense(input_shape[-1], name="query")
        self.k_dense = layers.Dense(input_shape[-1], name="key")
        self.v_dense = layers.Dense(input_shape[-1], name="value")
        self.o_dense = layers.Dense(input_shape[-1], name="combine_out")
        return super().build(input_shape)

    def call(self, inputs):
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        q = K.concatenate(tf.split(q, self.heads, axis=-1), axis=0)
        k = K.concatenate(tf.split(k, self.heads, axis=-1), axis=0)
        v = K.concatenate(tf.split(v, self.heads, axis=-1), axis=0)
        qk = tf.matmul(q, k, transpose_b=True)  
        qk = activations.softmax(qk / self.dk)
        qkv = tf.matmul(qk, v)
        qkv = K.concatenate(tf.split(qkv, self.heads, axis=0), axis=-1)
        return self.o_dense(qkv)

class MLPBlock(layers.Layer):
    """mlp block"""
    def __init__(self, liner_dim, dropout_rate, **kwargs):
        self.liner_dim = liner_dim
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dropout = layers.Dropout(self.dropout_rate)
        self.liner_1 = layers.Dense(self.liner_dim, activations.gelu)
        self.liner_2 = layers.Dense(input_shape[-1])
        return super().build(input_shape)

    def call(self, inputs):
        h = self.liner_1(inputs)
        h = self.dropout(h)
        h = self.liner_2(h)
        h = self.dropout(h)
        return h

class TransformerEncoder(layers.Layer):
    """transformerd encoder block"""
    def __init__(
        self, 
        liner_dim,
        atten_heads,
        dropout_rate,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.liner_dim = liner_dim
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):    
        self.multi_head_attens = MultiHeadAttention(
            name='multi_head_attention_layer',
            heads=self.atten_heads, 
            )
        self.mlp_block = MLPBlock(
            name='mlp_block_layer',
            liner_dim=self.liner_dim,
            dropout_rate = self.dropout_rate
            )
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.multi_head_attens(x)
        x = self.dropout(x)
        x = x + inputs
        y = self.layer_norm_2(x)
        y = self.mlp_block(y)
        return x + y

class ViT(models.Model):
	"""Implementation of VisionTransformer Based on Keras
	Args:
		- image_size: Input Image Size, integer or tuple
		- patch_size: Size of each patch, integer or tuple
		- num_classes: Number of output classes
		- hidden_dim: The embedding dimension of each patch, it should be set to an integer multiple of the `atten_heads`
		- mlp_dim: The projection dimension of the mlp_block, it is generally 4 times that of `hidden_dim`
		- atten_heads: Number of self attention heads
		- encoder_depth: Number of the transformer encoder layer
		- dropout_rate: Dropout probability
		- activatiion: Activation function of mlp_head
		- pre_logits: Insert pre_Logits layer or not 
		- include_mlp_head: Insert mlp_head layer or not
	"""
	def __init__(
		self, 
		image_size = 224,
		patch_size = 16,
		num_classes = 1000,
		hidden_dim = 768,
		mlp_dim = 3072,
		atten_heads = 12,
		encoder_depth = 12,
		dropout_rate = 0.,
		activation = "linear",
		pre_logits = True,
		include_mlp_head = True,
		**kwargs
		):
		assert isinstance(image_size, int) or isinstance(image_size, tuple), "`image_size` should be int or tuple !"
		assert isinstance(patch_size, int) or isinstance(patch_size, tuple), "`patch_size` should be int or tuple !"
		assert hidden_dim%atten_heads==0, "hidden_dim and atten_heads do not match !"

		if isinstance(image_size, int):
			image_size = (image_size, image_size)
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)

		size = "CUSTOM_SIZE"
		patch = f"{patch_size[0]}" if patch_size[0]==patch_size[1] else f"{patch_size[0]}_{patch_size[1]}"
		image = f"{image_size[0]}" if image_size[0]==image_size[1] else f"{image_size[0]}_{image_size[1]}"
		kwargs["name"] = f"ViT-{size}-{patch}-{image}"

		super().__init__(**kwargs)
		self.image_size = image_size[:2]
		self.patch_size = patch_size[:2]
		self.hidden_dim = hidden_dim
		self.mlp_dim = mlp_dim
		self.num_classes = num_classes
		self.atten_heads = atten_heads
		self.encoder_depth = encoder_depth
		self.dropout_rate = dropout_rate
		self.activation = activations.get(activation)
		self.pre_logits = pre_logits
		self.include_mlp_head = include_mlp_head
		self.load_weights_inf = None
		self.build((None, *self.image_size, 3))


	def build(self, input_shape):
		self.patch_embedding=PatchEmbedding(self.patch_size, self.hidden_dim, name="patch_embedding")
		self.add_cls_token = AddCLSToken(self.hidden_dim, name="add_cls_token")
		self.position_embedding = AddPositionEmbedding(name="position_embedding")
		self.encoder_blocks = [
			TransformerEncoder(
				self.mlp_dim, 
				self.atten_heads, 
				self.dropout_rate,
				name=f"transformer_block_{i}"
				) for i in range(self.encoder_depth)]
		self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
		self.extract_token = layers.Lambda(lambda x: x[:,0], name="extract_token")
		if self.pre_logits: self.pre_logits = layers.Dense(self.hidden_dim, activation="tanh", name="pre_logits")
		if self.include_mlp_head: self.mlp_head = layers.Dense(self.num_classes, self.activation, name="mlp_head")
		super().build(input_shape)
		self.call(layers.Input(input_shape[1:]))

	def call(self, inputs):
		x = self.patch_embedding(inputs)
		x = self.add_cls_token(x)
		x = self.position_embedding(x)
		for encoder in self.encoder_blocks:
			x = encoder(x)
		x = self.layer_norm(x)
		x = self.extract_token(x)
		if self.pre_logits:
			x = self.pre_logits(x)
		if self.include_mlp_head:
			x = self.mlp_head(x)
		return x

	def load_weights(self, *kwargs):
		super().load_weights(*kwargs)
		self.load_weights_inf = {l.name:"loaded - h5 weights" for l in self.layers}

	def loading_summary(self):
		""" Print model information about loading imagenet pre training weights.
		` - imagenet` means successful reading of imagenet pre training weights file
		` - not pre trained` means that the model did not load the imagenet pre training weights file
		` - mismatch` means that the pre training weights of the imagenet do not match the parameter shapes of the model layer
		` - not found` means that there is no weights for the layer in the imagenet pre training weights file
		` - h5 weights` means that the model is loaded with h5 weights
		"""
		if self.load_weights_inf:
			inf = [(key+' '*(35-len(key))+value+"\n") for (key,value) in self.load_weights_inf.items()]
		else:
			inf = [(l.name+' '*(35-len(l.name))+"not loaded - not pre trained"+"\n") for l in self.layers]
		print(f'Model: "{self.name}"')
		print("-"*65)
		print("layers" + " "*29 + "load weights inf")
		print("="*65)
		print("\n".join(inf)[:-1])
		print("="*65)

class Diffusion_ViT(models.Model):
	"""Implementation of VisionTransformer Based on Keras
	Args:
		- image_size: Input Image Size, integer or tuple
		- patch_size: Size of each patch, integer or tuple
		- hidden_dim: Dimension each patch is projected/embedded to, contextual info
		- mlp_dim: Dimension each patch is passed through in the transformer, libary of global facts
        - atten_heads: Number of heads in the transformer, splits the hidden dim into equal parts
		- encoder_depth: Number of the transformer encoder layer
		- dropout_rate: Dropout probability
		- pre_logits: Insert pre_Logits layer or not 
		- decoder: style of decoder
	"""
	def __init__(
		self, 
		image_size = 224,
		patch_size = 16,
		hidden_dim = 768,
		mlp_dim = 3072,
		atten_heads = 12,
		encoder_depth = 5,
		dropout_rate = 0.,
		pre_logits = False,
		decoder = 'linear',
		**kwargs
		):
		assert isinstance(image_size, int) or isinstance(image_size, tuple), "`image_size` should be int or tuple !"
		assert isinstance(patch_size, int) or isinstance(patch_size, tuple), "`patch_size` should be int or tuple !"
		assert hidden_dim%atten_heads==0, "hidden_dim and atten_heads do not match !"

		if isinstance(image_size, int):
			image_size = (image_size, image_size)
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)

		size = "CUSTOM_SIZE"
		patch = f"{patch_size[0]}" if patch_size[0]==patch_size[1] else f"{patch_size[0]}_{patch_size[1]}"
		image = f"{image_size[0]}" if image_size[0]==image_size[1] else f"{image_size[0]}_{image_size[1]}"
		kwargs["name"] = f"ViT-{size}-{patch}-{image}"

		super().__init__(**kwargs)
		self.image_size = image_size[:2]
		self.patch_size = patch_size[:2]
		self.hidden_dim = hidden_dim
		self.mlp_dim = mlp_dim

		self.atten_heads = atten_heads
		self.encoder_depth = encoder_depth
		self.dropout_rate = dropout_rate
		self.pre_logits = pre_logits
		self.decoder = decoder
		self.load_weights_inf = None
		self.build([(None, *self.image_size, 3), (None,1)])

	def create_conv_decoder(self, target_image_shape):
		"""Creates a sequential model to act as a convolutional decoder."""
		model = models.Sequential(name="conv_decoder")

		# input (None, 196, 768) 
		model.add(layers.Reshape((14, 14, 768)))  # Reshape to a 4D tensor to start the convolutional process
		model.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
		model.add(layers.BatchNormalization())
		
		# Upsample and convolve. Adjust the numbers and layers as per your requirements.
		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')) # 28 28 256
		model.add(layers.BatchNormalization())

		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')) # 56 56 128
		model.add(layers.BatchNormalization())

		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')) # 112 112 64

		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(3, kernel_size=3, padding='same',  activation='sigmoid')) # 224 224 3

		return model
	
	def create_linear_decoder(self, image_shape, patch_size):
		
		inputs = layers.Input((196, 768))
		x = layers.Dense(256*3, activation='relu')(inputs)
		x = layers.Reshape((224,224,3))(x)
		outputs = layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='sigmoid')(x)

	
		return Model(inputs=[inputs], outputs=[outputs], name='linear_decoder')

	def create_transformer_decoder(self, image_shape, patch_size):
		
		inputs = layers.Input((196, 768))
		x = layers.MultiHeadAttention(num_heads=self.atten_heads, key_dim=256)(inputs, inputs)
		x = layers.LayerNormalization()(x)
		x = layers.Dense(self.mlp_dim, activation='relu')(x)
		x = layers.Dense(256*3, activation='sigmoid')(x)
		outputs = layers.Reshape((224,224,3))(x)

	
		return Model(inputs=[inputs], outputs=[outputs], name='transformer_decoder')

	def build(self, input_shape):
		self.patch_embedding=PatchEmbedding(self.patch_size, self.hidden_dim, name="patch_embedding")
		self.add_cls_token = AddCLSToken_diffusion(self.hidden_dim, name="add_cls_token")
		self.position_embedding = AddPositionEmbedding(name="position_embedding")
		self.encoder_blocks = [
			TransformerEncoder(
				self.mlp_dim, 
				self.atten_heads, 
				self.dropout_rate,
				name=f"transformer_block_{i}"
				) for i in range(self.encoder_depth)]
		self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
		if self.pre_logits: self.pre_logits = layers.Dense(self.hidden_dim, activation="tanh", name="pre_logits")
		if self.decoder == "cnn": 
			self.decoder = self.create_conv_decoder(input_shape[0])
		elif self.decoder == 'transformer':
			self.decoder = self.create_transformer_decoder(input_shape[0], self.patch_size)
		elif self.decoder == 'linear':
			self.decoder = self.create_linear_decoder(input_shape[0], self.patch_size)	
		super().build(input_shape)
		self.call([layers.Input(shape=self.image_size + (3,)), layers.Input(shape=(1,))])

	def call(self, inputs):
		x, ts = inputs
		x = self.patch_embedding(x)
		x = self.add_cls_token([x, ts])
		x = self.position_embedding(x)

		x0 = self.encoder_blocks[0](x)
		x1 = self.encoder_blocks[1](x0)
		x2 = self.encoder_blocks[2](x1)
		
		x2 = layers.concatenate([x1,x2])
		x2 = layers.Dense(self.hidden_dim, activation='relu')(x2)
		x3 = self.encoder_blocks[3](x2)

		x3 = layers.concatenate([x3,x0])
		x3 = layers.Dense(self.hidden_dim, activation='relu')(x3)
		x = self.encoder_blocks[4](x3)

		x = self.layer_norm(x)
		x = x[:,2:,:]
		if self.pre_logits:
			x = self.pre_logits(x)
		x = self.decoder(x)
		return x

	def load_weights(self, *kwargs):
		super().load_weights(*kwargs)
		self.load_weights_inf = {l.name:"loaded - h5 weights" for l in self.layers}

	def loading_summary(self):
		""" Print model information about loading imagenet pre training weights.
		` - imagenet` means successful reading of imagenet pre training weights file
		` - not pre trained` means that the model did not load the imagenet pre training weights file
		` - mismatch` means that the pre training weights of the imagenet do not match the parameter shapes of the model layer
		` - not found` means that there is no weights for the layer in the imagenet pre training weights file
		` - h5 weights` means that the model is loaded with h5 weights
		"""
		if self.load_weights_inf:
			inf = [(key+' '*(35-len(key))+value+"\n") for (key,value) in self.load_weights_inf.items()]
		else:
			inf = [(l.name+' '*(35-len(l.name))+"not loaded - not pre trained"+"\n") for l in self.layers]
		print(f'Model: "{self.name}"')
		print("-"*65)
		print("layers" + " "*29 + "load weights inf")
		print("="*65)
		print("\n".join(inf)[:-1])
		print("="*65)

class ViT_multiword_AE(models.Model):
	"""Implementation of VisionTransformer Autoencoder using Keras
	Args:
		- image_size: Input Image Size
        - patch_size: Size of Image Patches
        - hidden_dim: Dimension each patch is projected/embedded to, contextual info
		- mlp_dim: Dimension each patch is passed through in the transformer, libary of global facts
        - atten_heads: Number of heads in the transformer, splits the hidden dim into equal parts
        - encoder_depth: Number of transformer encoders stacked
		- dropout_rate: Dropout probability
		- pre_logits: Insert pre_Logits layer or not
        - number_of_words: number of tokens (patches) in the latent space
        - decoder: style of decoder
        - encoder_only: True if used as encoder after training
	"""
	def __init__(
		self, 
		image_size,
		patch_size,
		hidden_dim,
		mlp_dim,
		atten_heads,
		encoder_depth,
		number_of_words,
		dropout_rate,
		pre_logits,
		decoder,
		encoder_only,
		**kwargs
		):
		assert isinstance(image_size, int) or isinstance(image_size, tuple), "`image_size` should be int or tuple !"
		assert isinstance(patch_size, int) or isinstance(patch_size, tuple), "`patch_size` should be int or tuple !"
		assert hidden_dim%atten_heads==0, "hidden_dim and atten_heads do not match !"

		if isinstance(image_size, int):
			image_size = (image_size, image_size)
		if isinstance(patch_size, int):
			patch_size = (patch_size, patch_size)

		size = "S"
		patch = f"{patch_size[0]}" if patch_size[0]==patch_size[1] else f"{patch_size[0]}_{patch_size[1]}"
		image = f"{image_size[0]}" if image_size[0]==image_size[1] else f"{image_size[0]}_{image_size[1]}"
		kwargs["name"] = f"ViT-{size}-{patch}-{image}"

		super().__init__(**kwargs)
		self.image_size = image_size[:2]
		self.patch_size = patch_size[:2]
		self.hidden_dim = hidden_dim
		self.mlp_dim = mlp_dim
		self.number_of_words = number_of_words

		self.atten_heads = atten_heads
		self.encoder_depth = encoder_depth
		self.dropout_rate = dropout_rate
		self.pre_logits = pre_logits
		self.decoder = decoder
		self.encoder_only = encoder_only
		self.load_weights_inf = None
		self.build((None, *self.image_size, 3))

	def create_conv_decoder(self, target_image_shape):
		"""Creates a sequential model to act as a convolutional decoder."""
		model = models.Sequential(name="conv_decoder")

		# input (None, number_of_words, 1024)
		model.add(layers.Dense(3136,activation='relu'))
		model.add(layers.Reshape((56, 56, self.number_of_words)))  # Reshape to a 4D tensor to start the convolutional process
		model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')) # 56 56 256
		model.add(layers.BatchNormalization())
		
		# Upsample and convolve. Adjust the numbers and layers as per your requirements.
		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')) # 112 112 128
		model.add(layers.BatchNormalization())

		model.add(layers.UpSampling2D(size=(2, 2)))
		model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')) # 224 224 64
		model.add(layers.Conv2D(3, kernel_size=3, padding='same',  activation='sigmoid')) # 224 224 3

		return model
	
	def create_linear_decoder(self, image_shape, patch_size):
		
		inputs = layers.Input((1, 768))
		x = layers.Dense(256*3, activation='relu')(inputs)
		x = layers.Reshape((224,224,3))(x)
		outputs = layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='sigmoid')(x)

	
		return Model(inputs=[inputs], outputs=[outputs], name='linear_decoder')

	def build(self, input_shape):
		self.patch_embedding=PatchEmbedding(self.patch_size, self.hidden_dim, name="patch_embedding")
		self.add_cls_token =  AddMultiCLSToken(self.hidden_dim, self.number_of_words, name="add_cls_token")
		self.position_embedding = AddPositionEmbedding(name="position_embedding")
		self.encoder_blocks = [
			TransformerEncoder(
				self.mlp_dim, 
				self.atten_heads, 
				self.dropout_rate,
				name=f"transformer_block_{i}"
				) for i in range(self.encoder_depth)]
		self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
		if not self.encoder_only:
			self.extract_token = layers.Lambda(lambda x: x[:,0:self.number_of_words], name="extract_token")
			if self.pre_logits: self.pre_logits = layers.Dense(self.hidden_dim, activation="tanh", name="pre_logits")
			if self.decoder == "cnn": 
				self.decoder = self.create_conv_decoder(input_shape[0])
			elif self.decoder == 'linear':
				self.decoder = self.create_linear_decoder(input_shape[0], self.patch_size)	
		super().build(input_shape)
		self.call(layers.Input(shape=self.image_size + (3,)))


	def call(self, inputs):
		x = self.patch_embedding(inputs)
		x = self.add_cls_token(x)
		x = self.position_embedding(x)

		for encoder in self.encoder_blocks:
			x = encoder(x)
		x = self.layer_norm(x)
		
		if not self.encoder_only:
			x = self.extract_token(x)

			if self.pre_logits:
				x = self.pre_logits(x)
			x = self.decoder(x)
		
		return x

	def load_weights(self, *kwargs):
		super().load_weights(*kwargs)
		self.load_weights_inf = {l.name:"loaded - h5 weights" for l in self.layers}

	def loading_summary(self):
		""" Print model information about loading imagenet pre training weights.
		` - imagenet` means successful reading of imagenet pre training weights file
		` - not pre trained` means that the model did not load the imagenet pre training weights file
		` - mismatch` means that the pre training weights of the imagenet do not match the parameter shapes of the model layer
		` - not found` means that there is no weights for the layer in the imagenet pre training weights file
		` - h5 weights` means that the model is loaded with h5 weights
		"""
		if self.load_weights_inf:
			inf = [(key+' '*(35-len(key))+value+"\n") for (key,value) in self.load_weights_inf.items()]
		else:
			inf = [(l.name+' '*(35-len(l.name))+"not loaded - not pre trained"+"\n") for l in self.layers]
		print(f'Model: "{self.name}"')
		print("-"*65)
		print("layers" + " "*29 + "load weights inf")
		print("="*65)
		print("\n".join(inf)[:-1])
		print("="*65)

def ViT_S8_AE(
	image_size=224,
	dropout_rate=0.1,
	pre_logits=None,
	number_of_words = 1,
	**kwargs
	):
	"""Implementation of ViT_S_Autoencoder Based on Keras.
	Args:
		- image_size: Input Image Size
        - patch_size: Size of Image Patches
        - hidden_dim: Dimension each patch is projected/embedded to, contextual info
		- mlp_dim: Dimension each patch is passed through in the transformer, libary of global facts
        - atten_heads: Number of heads in the transformer, splits the hidden dim into equal parts
        - encoder_depth: Number of transformer encoders stacked
		- dropout_rate: Dropout probability
		- pre_logits: Insert pre_Logits layer or not
        - number_of_words: number of tokens (patches) in the latent space
        - decoder: style of decoder
        - encoder_only: True if used as encoder after training
		

	return:
		- ViT_S8_AE: ViT with a patch_size of 8, hiddem_dim of 1024, 
		mlp_dim of 4096, atten_heads of 16, and encoder layers of 1
	"""
 
	vit = ViT_multiword_AE(
		image_size = image_size,
        patch_size=8,
        hidden_dim = 1024,
        mlp_dim = 4096,
        atten_heads = 16,
        encoder_depth = 1,
		dropout_rate=dropout_rate,
		pre_logits=pre_logits,
		number_of_words=number_of_words,
        decoder = 'cnn',
		encoder_only = False,
		**kwargs
		)


	return vit

