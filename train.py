import tensorflow as tf
import time
import argparse
import sys
import os
import tensorflow.contrib.slim.nets as nets



def inputs(dataset,img_size,batch_size):
  feature={'image':tf.FixedLenFeature([],tf.string),'label':tf.FixedLenFeature([],tf.int64)}
  # Create a list of filenames and pass it to a queue
  filename_queue = tf.train.string_input_producer([dataset], num_epochs=FLAGS.num_epochs)
  # Define a reader and read the next record
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  # Decode the record read by the reader
  features = tf.parse_single_example(serialized_example, features=feature)
  # Convert the image data from string back to the numbers
  image = tf.decode_raw(features['image'], tf.float32)
  
  # Cast label data into int32
  label = tf.cast(features['label'], tf.int32)
  # Reshape image data into the original shape
  image = tf.reshape(image, [32, 32, 1])
  
  # Creates batches by randomly shuffling tensors
  images, labels = tf.train.shuffle_batch([image, label], batch_size=5, capacity=600, num_threads=10, min_after_dequeue=1)
  return images,labels





def cnn_model_fn(features,labels):
  #Input Layer
  input_layer=tf.reshape(features['images'],[5,32,32,1])

  #Convolution Layer 1
  conv1=tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[2,2],padding="same",activation=tf.nn.relu)

  #Pooling Layer #1
  pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

  #Convolutional Layer #2 and Pooling Layer #2
  conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[2,2],padding="same",activation=tf.nn.relu)

  #Pooling Layer #2
  pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)

  #Dense Layer
  pool2_shape=pool2.get_shape()
  pool2_features=pool2_shape[1:4].num_elements()
  pool2_flat=tf.reshape(pool2,[5,pool2_features])
  dense=tf.layers.dense(inputs=pool2_flat,units=100,activation=tf.nn.relu)
  dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=True )

  #Logits Layer
  logits=tf.layers.dense(inputs=dropout,units=5)


  prediction={
  "classes":tf.argmax(input=logits,axis=1),"probabilities":tf.nn.softmax(logits,name="softmax_tensor")
    }

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    
  return prediction,loss





def run_training(start_time):
  slim=tf.contrib.slim
  with tf.Graph().as_default():
      images_placeholder=tf.placeholder(tf.float32,shape=(FLAGS.batch_size,FLAGS.img_size,1),name='image')
      labels_placeholder=tf.placeholder(tf.float32,shape=(FLAGS.batch_size),name='labels')

      images,labels=inputs(FLAGS.train_data,FLAGS.img_size,FLAGS.batch_size)

      images_placeholder=images
      labels_placeholder=labels
      tf.summary.image('input',images_placeholder,1)

      features={'images':images_placeholder,'labels':labels_placeholder}

      prediction,loss=cnn_model_fn(features,labels)
      accuracy=tf.metrics.accuracy(labels=labels, predictions=prediction["classes"])
      accuracy=tf.reduce_mean(accuracy)

      tf.summary.scalar('loss',loss)
      tf.summary.scalar('accuracy_mean',accuracy)

      #Optimizer
      #optimizer=tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      #optimizer=tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
      #optimizer=tf.train.ProximalGradientDescentOptimizer(learning_rate=FLAGS.learning_rate) 
      train_op=slim.learning.create_train_op(loss,optimizer)
      tf.summary.scalar('learning_rate',FLAGS.learning_rate)
      init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

      #create checkpoint
      saver=tf.train.Saver(max_to_keep=10)

      #Summary Tensor
      summary_op=tf.summary.merge_all()

      #Instantiate a SummaryWriter to output summaries and the graph
      summary_writer=tf.summary.FileWriter(FLAGS.log_dir,tf.get_default_graph())

      loss_sum=0
      acc_sum=0
      batch_sum=0
     

      with tf.Session() as sess:
        sess.run(init_op)
        #Start the input thread
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)



        for step in range(FLAGS.max_steps):

          _,loss_value,acc_value=sess.run([train_op,loss,accuracy])
          print("loss",loss_value)
          print("accuracy value",acc_value)

          loss_sum += loss_value * FLAGS.batch_size
          acc_sum += acc_value * FLAGS.batch_size
          batch_sum += FLAGS.batch_size
          #Print an overview
          if step % 100 == 0:
            duration = time.time() - start_time
            print('Step %d: loss = %.4f, accuracy = %.4f (%.3f sec)' % 
                  (step, loss_sum/batch_sum, acc_sum/batch_sum, duration))

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            loss_sum = 0
            acc_sum = 0
            batch_sum =0
            start_time = time.time()
                    

                # Save a checkpoint and evaluate the model periodically
          if (step + 1)%1000==0 or (step+1)==FLAGS.max_steps:
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step = step)
            # Evaluate against the training set
            pass

          print('Done training for %d steps.' % (FLAGS.max_steps))
          # When done, ask the threads to stop.
          coord.request_stop()
          # Wait for threads to finish.
          coord.join(threads)

                 
    





def main(_):
  start_time=time.time()
  run_training(start_time)





if __name__ == '__main__':
    datapath='/home/robinreni/Documents/pythonprojects/cnn/christams/christmas1_temp.tfrecords'

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run trainer')
    parser.add_argument('--max_steps', type=int, default=200, help='Number of steps to run trainer')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--img_size', type=int, default=1024, help='Image witdh and height')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--train_data', type=str, default=datapath, help='TFRecords filename of training data')
    parser.add_argument('--log_dir', type=str, default='./log/test', help='Directory to put the log data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()