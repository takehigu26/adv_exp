import tensorflow as tf
from tensorflow import keras

class BaseModel(tf.keras.Model):
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
        super(AdversarialModel, self).__init__(*args[2:], **kwargs)
        self.explanation_loss = tf.Variable(0.)
        self.targets = args[0]
        self.alpha = args[1]
    
    def train_step(self, data):
        X, y = data
        with tf.GradientTape(persistent=True) as t:
            t.watch(X)
            y_pred = self(X, training=True)
            performance_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #tf.print(performance_loss.shape)
            explanation_loss_entire = t.gradient(performance_loss, X)
            self.explanation_loss.assign(0.)
            for target in self.targets:
                explanation_loss_r = explanation_loss_entire[:, target[0]]
                explanation_loss_r = tf.norm(explanation_loss_r, 1)
                explanation_loss_r = explanation_loss_r * target[1]
                self.explanation_loss.assign_add(explanation_loss_r)
            #tf.print(tf.norm(self.explanation_loss, 1))
            #tf.print(self.explanation_loss)
            exp_term = self.alpha * self.explanation_loss / X.shape[0]
            #tf.print(tf.norm(exp_term, 1))
            total_loss = performance_loss + exp_term
            #tf.print(tf.norm(total_loss, 1))
        grads = t.gradient(total_loss, self.trainable_variables)
        del t
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    