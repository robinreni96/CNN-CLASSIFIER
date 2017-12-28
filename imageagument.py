import tensorflow as tf 
#This code aguments the image into various shapes,size and structure.
def agument(images,labels,label,image_array,augementation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
	if use_random_rotation:
		#Rotate the image randomly and save it
		image=tf.contrib.keras.preprocessing.image.random_rotation(image_array, 20, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)
	
	if use_random_shear:
		#Shear the image randomly and save it
		image=tf.contrib.keras.preprocessing.image.random_shear(image_array, 0.2, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)
		
	if use_random_shift:
		#Shift the image to a position and save it
		image=tf.contrib.keras.preprocessing.image.random_shift(image_array, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=0)
		images.append(image)
		labels.append(label)	

	#Return the images and labels in a list format
	return images,labels
