from tensorflow.keras import backend as K

def WeightedCCE(class_weights, alpha=1.0):
    """
    Cross entropy loss weighted by class weights and scaling term alpha
    modified from https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    class_weights = K.variable(class_weights)
    a = K.variable(alpha)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * class_weights
        loss = -K.sum(loss, -1)
        return a*loss

    return loss