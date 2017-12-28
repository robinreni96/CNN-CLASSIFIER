import tensorflow as tf 






def agument(images,labels,label,image_array,augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	if use_random_rotation:
		image=tf.contrib.keras.preprocessing.image.random_rotation(image_array, 20, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)
	if use_random_shear:
		image=tf.contrib.keras.preprocessing.image.random_shear(image_array, 0.2, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)
	if use_random_shift:
		image=tf.contrib.keras.preprocessing.image.random_shift(image_array, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)	

	return images,labels