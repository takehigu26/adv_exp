import tensorflow as tf

def no_attack(model, X, y, **kwargs):
    delta = tf.Variable(tf.zeros_like(X.shape[1:], dtype="float32"))
    return delta

def epoch_explanation(loader, model, attack, sensitive_feature_id, norm=1, normalise=True, e_alpha=0.25, optimizer=None,
                      **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    loss = tf.keras.losses.categorical_crossentropy
    n_batch = 0
    for X, y in loader:
        #delta = attack(model, X, y, **kwargs)
        delta = no_attack(model, X, y, **kwargs)
        if optimizer:
            with tf.GradientTape(persistent=True) as t:
                t.watch(delta)
                t.watch(X)
                yp = model(X + delta)

                performance_loss = loss(y_true=y, y_pred=yp, from_logits=False)
                l = model.layers[-1]
                l.activation = tf.keras.activations.linear
                # TODO for CNN

                # TODO X or X + delta?
                explanation = t.gradient(performance_loss, [X])[0][:, sensitive_feature_id]
                # define the explanation loss to be L1 norm over the gradient of the sensitive feature of the input
                # L1 norm because we want to induce sparsity (this is a guess)
                # L1 norm makes is important when there is a differnce whether x is exactly 0 or not
                explanation_loss = tf.norm(explanation, norm)
                if normalise:
                    explanation_loss = tf.compat.v2.math.divide(explanation_loss, tf.cast(X.shape[0], "float32"))

                    # L = original_loss + \alpha * explanatoin_loss
                total_loss = performance_loss + e_alpha * explanation_loss

                # check position
                l.activation = tf.keras.activations.softmax

            # optimize with an optimizer
            grads = t.gradient(total_loss, [*model.weights])
            optimizer.apply_gradients(zip(grads, model.weights))

        loss_val, err, = model.evaluate(X + delta, y, verbose=0)
        err = 1 - err
        total_err += err
        total_loss += loss_val
        n_batch += 1
    return total_err / n_batch, total_loss / n_batch