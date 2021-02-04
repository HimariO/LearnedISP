import tensorflow as tf
import tensorflow_addons as tfa


class WeightNormalization(tfa.layers.WeightNormalization):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    """

    def __init__(self, layer, data_init=True, inference=False, **kwargs):
        super(WeightNormalization, self).__init__(
            layer, data_init=data_init, **kwargs)
        self.inference_mode = inference

    def call(self, inputs):
        """Call `Layer`"""
        if self.inference_mode:
            self.layer.kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g
            return self.layer(inputs)
        else:
            def _do_nothing():
                return tf.identity(self.g)

            def _update_weights():
                # Ensure we read `self.g` after _update_weights.
                with tf.control_dependencies(self._initialize_weights(inputs)):
                    return tf.identity(self.g)

            g = tf.cond(self._initialized, _do_nothing, _update_weights)

            with tf.name_scope('compute_weights'):
                # Replace kernel by normalized weight variable.
                self.layer.kernel = tf.nn.l2_normalize(
                    self.v, axis=self.kernel_norm_axes) * g

                # Ensure we calculate result after updating kernel.
                update_kernel = tf.identity(self.layer.kernel)
                with tf.control_dependencies([update_kernel]):
                    outputs = self.layer(inputs)
                    return outputs