import tensorflow as tf
import tensorflow as tf
import os 
import PIL
import imageagument

global tk
tk=tf.contrib.keras.preprocessing

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_preprocess(image_path,label_name):
	images=[]
	labels=[]
	image=tk.image.load_img(image_path,grayscale=True,target_size=(32,32))
	image_array=tk.image.img_to_array(image,data_format=None)
	images.append(image_array)

	label=image_path.split(os.path.sep)[-2]
	if label == 'cake':
		label=0
	elif label == 'cards':
		label=1
	elif label == 'santa':
		label=2
	elif label == "star":
		label=3
	elif label == "tree":
		label=4
	labels.append(label)
	images,labels=imageagument.agument(images,labels,label,image_array)
	return images,labels




path="""Directory path to the image set""" 
files=os.listdir(path)
tfrecord_filename='christmas.tfrecords'
writer=tf.python_io.TFRecordWriter(tfrecord_filename)
for file in files:
	temp_path=path+'/'+file
	list_images=os.listdir(temp_path)
	for i in list_images:
		image_path=temp_path+'/'+i
		images,labels=image_preprocess(image_path,file)
		for i in range(len(labels)):
			feature={'label':_int64_feature(labels[i]),
		          'image':_bytes_feature(images[i].tostring())}
			example=tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())
writer.close()
