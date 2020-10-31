class YourModel(Model):
    def __init__(self):
        super(YourModel, self).__init__()

        # define your model architecture here as an attribute of the class

  def train_step(data):
      with tf.GradientTape() as tape:
          # foward pass data through the architecture
          # compute loss (y_true, y_pred, any other param)
      
      # weight update
      gradients = tape.gradient(loss, self.trainable_variables) 
      self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

      return {
          'loss': loss
          # other losses
      }