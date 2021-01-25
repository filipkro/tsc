import tensorflow as tf
import numpy as np
# import tensorflow.keras.backend as K

# This function to generate evidence is used for the first example
def relu_evidence(logits):
    return tf.nn.relu(logits)

# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits):
    return tf.exp(tf.clip_by_value(logits,-10,10))

# This one is another alternative and
# usually behaves better than the relu_evidence
def softplus_evidence(logits):
    return tf.nn.softplus(logits)

def KL(alpha):
    K=3
    beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)

    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)

    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni

    return kl

def mse_loss(p, alpha, global_step, annealing_step):
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((p-m)**2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))

    alp = E*(1-p) + 1
    C =  annealing_coef * KL(alp)
    return (A + B) + C

def evidence_loss(y_true, logits):
    evidence = softplus_evidence(logits)
    alpha = evidence + 1
    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    E = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((y_true-m)**2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    return (A + B)

    # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    #
    #annealing_coef = 0.0
    #alp = E*(1-y_true) + 1
    #C =  annealing_coef * KL(alp)
    #return (A + B) + C
    return (A + B)

def get_evidence(logits):
    evidence = softplus_evidence(logits)
    return evidence

def get_uncertainty(logits):
    K = 3
    evidence = get_evidence(logits)
    alpha = evidence + 1
    u = K / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty

    return u

def get_prob(logits):
    evidence = get_evidence(logits)
    alpha = evidence + 1
    prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True)

    return prob

# def metric_prob
