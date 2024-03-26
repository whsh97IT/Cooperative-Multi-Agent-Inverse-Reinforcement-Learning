import numpy as np

class BaseOptimizer:
    def __init__(self):
        self.params = None

    def reset(self, params):
        self.params = params

    def step(self, gradient):
        raise NotImplementedError

    def normalize(self, norm_order=None):
        return GradientNormalizer(self, norm_order)


class GradientAscent(BaseOptimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = 0

    def reset(self, params):
        super().reset(params)
        self.iterations = 0

    def step(self, gradient):
        lr = self.learning_rate(self.iterations) if callable(self.learning_rate) else self.learning_rate
        self.iterations += 1
        self.params += lr * gradient


class ExpGradientAscent(BaseOptimizer):
    def __init__(self, learning_rate, normalize=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.iterations = 0

    def reset(self, params):
        super().reset(params)
        self.iterations = 0

    def step(self, gradient):
        lr = self.learning_rate(self.iterations) if callable(self.learning_rate) else self.learning_rate
        self.iterations += 1
        self.params *= np.exp(lr * gradient)
        if self.normalize:
            self.params /= np.sum(self.params)


class GradientNormalizer(BaseOptimizer):
    def __init__(self, optimizer, norm_order=None):
        super().__init__()
        self.optimizer = optimizer
        self.norm_order = norm_order

    def reset(self, params):
        super().reset(params)
        self.optimizer.reset(params)

    def step(self, gradient):
        normalized_gradient = gradient / np.linalg.norm(gradient, self.norm_order)
        self.optimizer.step(normalized_gradient)


def lr_linear_decay(initial_lr=0.2, decay=1.0, steps=1):
    def lr(k):
        return initial_lr / (1.0 + decay * np.floor(k / steps))
    return lr

def lr_power_decay(initial_lr=0.2, decay=1.0, steps=1, power=2):
    def lr(k):
        return initial_lr / (1.0 + decay * np.floor(k / steps))**power
    return lr

def lr_exponential_decay(initial_lr=0.2, decay=0.5, steps=1):
    def lr(k):
        return initial_lr * np.exp(-decay * np.floor(k / steps))
    return lr

class Init:
    def __init__(self):
        pass

    def initialize(self, shape):
        raise NotImplementedError

    def __call__(self, shape):
        return self.initialize(shape)

class UniformInit(Init):
    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self.low = low
        self.high = high

    def initialize(self, shape):
        return np.random.uniform(low=self.low, high=self.high, size=shape)

class ConstantInit(Init):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        return np.full(shape, self.value if not callable(self.value) else self.value(shape))
