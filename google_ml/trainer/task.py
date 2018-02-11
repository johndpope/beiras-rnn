import argparse
import glob
import json
import os

import trainer.model as model
from tensorflow.python.client import device_lib

from keras.models import load_model
import model
import keras as keras
from tensorflow.python.lib.io import file_io
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf

WINDOWS_SIZE = 100
NUM_CHARS = 55
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
BEIRAS_MODEL = 'beiras.hdf5'
# In local, use a value to be able to load train data in memory
#CHUNK_SIZE = 5000
# In Cloud, use this value
CHUNK_SIZE = None

class ContinuousEval(keras.callbacks.Callback):
  """Continuous eval callback to evaluate the checkpoint once
     every so many epochs.
  """

  def __init__(self,
               eval_frequency,
               eval_files,
               learning_rate,
               job_dir,
               steps=0):
    self.eval_files = eval_files
    self.eval_frequency = eval_frequency
    self.learning_rate = learning_rate
    self.job_dir = job_dir
    self.steps = steps

  def on_epoch_begin(self, epoch, logs={}):
    if epoch > 0 and epoch % self.eval_frequency == 0:

      # Unhappy hack to work around h5py not being able to write to GCS.
      # Force snapshots and saves to local filesystem, then copy them over to GCS.
      model_path_glob = 'checkpoint.*'
      if not self.job_dir.startswith("gs://"):
        model_path_glob = os.path.join(self.job_dir, model_path_glob)
      checkpoints = glob.glob(model_path_glob)
      if len(checkpoints) > 0:
        checkpoints.sort()
        beiras_model = load_model(checkpoints[-1])
        beiras_model = model.compile_model(beiras_model, self.learning_rate)
        x_eval, y_eval = model.get_array_x_y(self.eval_files, self.steps, WINDOWS_SIZE, NUM_CHARS)
        loss, acc =beiras_model.evaluate(x_eval,y_eval)
        print '\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
            epoch, loss, acc, beiras_model.metrics_names)
        if self.job_dir.startswith("gs://"):
          copy_file_to_gcs(self.job_dir, checkpoints[-1])
      else:
        print '\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch)


def dispatch(train_files,
             eval_files,
             job_dir,
             train_steps,
             eval_steps,
             train_batch_size,
             eval_batch_size,
             learning_rate,
             eval_frequency,
             eval_num_epochs,
             num_epochs,
             checkpoint_epochs,
             gpus):

    #With severals GPU you use 2 models, un for training and other for store.
    # The first one is assigned to the CPU,
    # the other run in the GPU and is generated using multi_gpu_model


  if gpus <= 1:
    model_train = model.model_fn(NUM_CHARS,window_size=WINDOWS_SIZE)
    model_save = model_train
  else:
    with tf.device("/cpu:0"):
      model_save = model.model_fn(NUM_CHARS, window_size=WINDOWS_SIZE)
    model_train = multi_gpu_model(model_save, gpus=gpus)
    model.compile_model(model_save, learning_rate)
    print(model_save.summary())
  model.compile_model(model_train, learning_rate)
  print(model_train.summary())

  try:
    os.makedirs(job_dir)
  except:
    pass

  # Unhappy hack to work around h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  checkpoint_path = FILE_PATH
  if not job_dir.startswith("gs://"):
    checkpoint_path = os.path.join(job_dir, checkpoint_path)

  # Model checkpoint callback
  checkpoint = keras.callbacks.ModelCheckpoint(
      checkpoint_path,
      monitor='val_loss',
      verbose=0,
      period=checkpoint_epochs,
      mode='max')

  # Continuous eval callback
  evaluation = ContinuousEval(eval_frequency,
                              eval_files,
                              learning_rate,
                              job_dir,
                              steps=eval_steps)

  # Tensorboard logs callback
  tblog = keras.callbacks.TensorBoard(
      log_dir=os.path.join(job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

  callbacks=[checkpoint, evaluation, tblog]

  x,y=model.get_array_x_y(train_files, train_steps, WINDOWS_SIZE,NUM_CHARS)
  
  model_train.fit(x,y,epochs=num_epochs,callbacks=callbacks,batch_size=500)

  # Unhappy hack to work around h5py not being able to write to GCS.
  # Force snapshots and saves to local filesystem, then copy them over to GCS.
  if job_dir.startswith("gs://"):
    model_save.save(BEIRAS_MODEL)
    copy_file_to_gcs(job_dir, BEIRAS_MODEL)
  else:
      model_save.save(os.path.join(job_dir, BEIRAS_MODEL))

  # Convert the Keras model to TensorFlow SavedModel
  model.to_savedmodel(model_save, os.path.join(job_dir, 'export'))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
  with file_io.FileIO(file_path, mode='r') as input_f:
    with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
        output_f.write(input_f.read())


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train-files',
                      required=True,
                      type=str,
                      help='Training files local or GCS', nargs='+')
  parser.add_argument('--eval-files',
                      required=True,
                      type=str,
                      help='Evaluation files local or GCS', nargs='+')
  parser.add_argument('--job-dir',
                      required=True,
                      type=str,
                      help='GCS or local dir to write checkpoints and export model')
  parser.add_argument('--train-steps',
                      type=int,
                      default=0,
                      help="""\
                       Maximum number of training steps to perform. This is \
                       number of train data to use. \
                       Default:0 .- use all train file
                       
                      """)
  parser.add_argument('--eval-steps',
                      help='Number of steps to run evalution for at each checkpoint',
                      default=0,
                      type=int)
  parser.add_argument('--train-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for training steps')
  parser.add_argument('--eval-batch-size',
                      type=int,
                      default=40,
                      help='Batch size for evaluation steps')
  parser.add_argument('--learning-rate',
                      type=float,
                      default=0.003,
                      help='Learning rate for SGD')
  parser.add_argument('--eval-frequency',
                      default=10,
                      type=int,
                      help='Perform one evaluation per n epochs')
  parser.add_argument('--eval-num-epochs',
                     type=int,
                     default=1,
                     help='Number of epochs during evaluation')
  parser.add_argument('--num-epochs',
                      type=int,
                      default=20,
                      help='Maximum number of epochs on which to train')
  parser.add_argument('--checkpoint-epochs',
                      type=int,
                      default=5,
                      help='Checkpoint per n training epochs')
  parser.add_argument('--gpus',
                      type=int,
                      default=1,
                      help='number gpus')


  parse_args, unknown = parser.parse_known_args()
  print(device_lib.list_local_devices())
  dispatch(**parse_args.__dict__)
