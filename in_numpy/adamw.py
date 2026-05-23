import numpy as np


def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g**2

    m_corrected = m_new / (1 - beta1**t)
    v_corrected = v_new / (1 - beta2**t)

    w -= lr * weight_decay * w
    w -= lr * m_corrected / (np.sqrt(v_corrected) + epsilon)
    return w, m_new, v_new


if __name__ == "__main__":
    import numpy as np

    w = np.array([1.0, 2.0])
    g = np.array([0.1, -0.2])
    m = np.zeros(2)
    v = np.zeros(2)
    t = 1
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 0.1
    w_new, m_new, v_new = adamw_update(
        w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay
    )
    print(np.round(w_new, 4))
