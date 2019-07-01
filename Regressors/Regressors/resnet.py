# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import pandas as pd 
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 

from utils.utils import save_logs

class Classifier_RESNET: 

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True):
		self.output_directory =str( output_directory)
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')
        
        
        
        

	def build_model(self, input_shape, nb_classes):
		n_feature_maps =32

		input_layer = keras.layers.Input(shape=input_shape)
		
		# BLOCK 1 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_1 = keras.layers.add([shortcut_y, conv_z])
		output_block_1 = keras.layers.Activation('relu')(output_block_1)

		# BLOCK 2 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# expand channels for the sum 
		shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
		shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

		output_block_2 = keras.layers.add([shortcut_y, conv_z])
		output_block_2 = keras.layers.Activation('relu')(output_block_2)

		# BLOCK 3 

		conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = keras.layers.Activation('relu')(conv_x)

		conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = keras.layers.Activation('relu')(conv_y)

		conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		# no need to expand channels because they are equal 
		shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

		output_block_3 = keras.layers.add([shortcut_y, conv_z])
		output_block_3 = keras.layers.Activation('relu')(output_block_3)

		# FINAL 
		
		gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
            	


		output_layer = keras.layers.Dense(nb_classes, activation='linear')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='mse', optimizer=keras.optimizers.Adam())

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path = self.output_directory + 'best_model.hdf5' 

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model
	
	def fit(self, x_train, y_train, x_val, y_val, x_true,y_true): 
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 128
		nb_epochs =  40

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))


		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs
		                      , validation_data=(x_val,y_val), callbacks=self.callbacks)


		model = keras.models.load_model(self.output_directory+'best_model.hdf5')
 		
        
		predicted       = model.predict(x_true)
		print(predicted)
		
		#predicted = np.exp(predicted)
		#predict_label = np.exp(predict_label)
		print('mean_predicted: %d' % (np.mean(predicted)))
		print('std: %d' % (np.std(predicted)))
		
		predicted       = np.exp(predicted)
        
		diff 	  = (predicted - y_true)
		rel_diff  = diff/(predicted + y_true)
		
		
		print(rel_diff, np.min(rel_diff) , np.max(rel_diff))
		
		
		mean_diff = np.mean(rel_diff)
		
		std = np.std(rel_diff)
		
		
		print('std: %d' %std)
		print('mean %d' %mean_diff)
		#can modell predict unnoised data
		#no_noise = (predicted_no_noise - predict_label)/predicted_no_noise
		
		abs_rel_diff = np.abs(rel_diff)
		maximum = np.max(abs_rel_diff)
		
		#logbinning =  10**np.linspace(np.log10(1e-10), np.log10(maximum) , 100)
		bad_points = []
		for i,j in zip(rel_diff, y_true):
			if np.abs(i) > 0.5 :
				#print(j)
				bad_points.append(j)
		
		print(len(bad_points))
		history = np.genfromtxt(self.output_directory+'history.csv', delimiter=',', names=True) 
		fig, ax = plt.subplots(1)
		ax.plot(history['epoch'], history['loss'],     label='training')
		ax.plot(history['epoch'], history['val_loss'], label='validation')
		ax.legend()
		ax.set(xlabel='epoch', ylabel='loss')
		plt.savefig('%s/loss.png'%(path_outfile))
		plt.close()
		
		plt.subplot(211)
		plt.hist(rel_diff, bins=50)
		plt.title('(e_predict - e_real)/(e_predict+e_real)')
		plt.yscale('log')
		
		plt.subplot(212)
		plt.hist(diff, bins=50)
		plt.title('(e_predict - e_real)')
		plt.yscale('log')
		plt.savefig('%s/acc.png'%(path_outfile))
		plt.close()		
		
		save_logs(self.output_directory, hist, y_pred, y_true)

		keras.backend.clear_session()

		
