import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

class BaseModel(tf.keras.Model):
    '''
    def __init__(self, n_features):
        super(BaseModel, self).__init__()
        self.inputs_layer = keras.layers.Input(shape=n_features)
        self.layer1 = keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))
        self.layer2 = keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))
        self.layer3 = keras.layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.03))
        self.outputs_layer = keras.layers.Dense(2, activation=tf.nn.softmax)
        
    def call(self, inputs, training=False):
        x = self.inputs_layer(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.outputs_layer(x)
        return x
       '''
    def train_step(self, data):
        X, y = data
        with tf.GradientTape(persistent=True) as t:
            y_pred = self(X, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = t.gradient(loss, self.trainable_variables)
        del t
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class AdversarialModel(tf.keras.Model):
    def __init__(self,  *args, **kwargs):
        super(AdversarialModel, self).__init__(*args, **kwargs)
        self.explanation_loss = tf.Variable(0.)
        
    def train_step(self, data):
        alpha, targets = 0.1, [[1, 1], [15, -1]]
        X, y = data
        with tf.GradientTape(persistent=True) as t:
            t.watch(X)
            y_pred = self(X, training=True)
            performance_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            explanation_loss_entire = t.gradient(performance_loss, X)
            tf.zeros_like(self.explanation_loss)
            #print(explanation_loss_entire)
            for target in targets:
                explanation_loss_r = explanation_loss_entire[:, target[0]]
                explanation_loss_r = tf.norm(explanation_loss_r, 1)
                explanation_loss_r = explanation_loss_r * target[1]
                self.explanation_loss.assign_add(explanation_loss_r)
                #explanation_loss += explanation_loss_r
            #print("explanation_loss: "+str(tf.norm(self.explanation_loss, 2)))
            total_loss = performance_loss + (alpha * self.explanation_loss / X.shape[0])
        grads = t.gradient(total_loss, self.trainable_variables)
        del t
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}