import numpy as np
from collections import OrderedDict


class GradientDescent(object):
    def __init__(self, lr=0.0001):
        self.lr = lr

    def update(self, model, grad_dict):
        for l_name, layer in model.layers.items():
            for param_name, param in layer.params.items():
                grad = grad_dict[l_name][param_name]
                model.layers[l_name].params[param_name] = self._calc_next_param(param, grad)

    def _calc_next_param(self, param, grad):
        return param - self.lr * grad


class MomentumSGD(object):
    def __init__(self, lr=0.0001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.w = None

    def update(self, model, grad_dict):
        # 初回時の w はゼロにする
        if self.w is None:
            self._init_w(model)

        # 更新
        for l_name, layer in model.layers.items():
            for param_name, param in layer.params.items():
                grad = grad_dict[l_name][param_name]
                w = self.w[l_name][param_name]

                # パラメータ更新
                model.layers[l_name].params[param_name] = self._calc_next_param(param, grad, w)

                # wを更新
                self.w[l_name][param_name] = self._calc_next_w(grad, w)

    def _calc_next_param(self, param, grad, w):
        return param + self._calc_next_w(grad, w)

    def _calc_next_w(self, grad, w):
        return - self.lr * grad + self.momentum * w

    def _init_w(self, model,):
        self.w = OrderedDict()
        for l_name, layer in model.layers.items():
            self.w[l_name] = OrderedDict()
            for param_name, _ in layer.params.items():
                self.w[l_name][param_name] = 0.0


class Adam(object):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = None
        self.v = None
        
        self.iter = 0
        
    def update(self, model, grad_dict):
        # 初回時の m, v はゼロにする
        if self.m is None and self.v is None:
            self._init_mv(model)
        
         # 更新
        self.iter += 1
        for l_name, layer in model.layers.items():
            for param_name, param in layer.params.items():
                grad = grad_dict[l_name][param_name]
                self.m[l_name][param_name] = self.beta1 * self.m[l_name][param_name] + (1 - self.beta1) * grad
                self.v[l_name][param_name] = self.beta2 * self.v[l_name][param_name] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[l_name][param_name] / (1 - self.beta1 ** self.iter)
                v_hat = self.v[l_name][param_name] / (1 - self.beta2 ** self.iter)
                
                model.layers[l_name].params[param_name] = self._calc_next_param(param, m_hat, v_hat)
                
    def _calc_next_param(self, param, m_hat, v_hat):
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
    def _init_mv(self, model):
        self.m = OrderedDict()
        self.v = OrderedDict()
        for l_name, layer in model.layers.items():
            self.m[l_name] = OrderedDict()
            self.v[l_name] = OrderedDict()
            for param_name, _ in layer.params.items():
                self.m[l_name][param_name] = 0.0
                self.v[l_name][param_name] = 0.0
