import math
from typing import Union, Collection

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float]

class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)
        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function
    def loss(self, data: TFData):
        # losses = []
        # for k in range(self.K):
        #     log_normal = self.neglog_normal_pdf(data, self.mean[k], self.logvar[k])
        #     log_prob_k = self.logpi[k] - log_normal
        #     losses.append(log_prob_k)
        # return -tf.reduce_logsumexp(tf.stack(losses, axis=-1), axis=-1)
        neglog_likelihood = -self.neglog_normal_pdf(data, self.mean, self.logvar)
        loss = -tf.reduce_logsumexp(self.logpi + neglog_likelihood, axis=-1)
        return loss

    def p_xz(self, x: TFData, k: int) -> TFData:
        # log_normal = self.neglog_normal_pdf(x, self.mean[k], self.logvar[k])
        # return self.logpi[k] - log_normal
        # component_pdf = dist.normal_pdf(x, dist.mu[k], dist.sigma2[k])
        # return dist.pi[k] * component_pdf
        mean_k = self.mean[k]
        logvar_k = self.logvar[k]

        return tf.exp(-self.neglog_normal_pdf(x, mean_k, logvar_k))

    def p_x(self, x: TFData) -> TFData:
        # log_prob = []
        # for k in range(self.K):
        #     log_normal = self.neglog_normal_pdf(x, self.mean[k], self.logvar[k])
        #     log_prob_k = self.logpi[k] - log_normal
        #     log_prob.append(log_prob_k)
        # return tf.reduce_logsumexp(tf.stack(log_prob, axis=-1), axis=-1)
        # log_prob_k = []
        # for k in range(self.K):
        #     log_prob_k.append(tf.math.log(self.p_xz(x, k)))
        # return tf.reduce_logsumexp(log_prob_k)
        probs = [tf.exp(self.logpi[k]) * self.p_xz(x, k) for k in range(self.K)]
        return tf.reduce_sum(probs, axis=0)
