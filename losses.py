import numpy as np
from collections import OrderedDict


class BaseLoss(object):
    def __init__(self):
        self.eps = 0.001
        self._loss = 0.0

    def __call__(self, model, inputs, targets, is_grad=True):
        # 損失を計算
        self._loss = np.mean([self._calc_loss(model.forward(_input), target) for _input, target in zip(inputs, targets)])
        
        if is_grad:
            grad_dict = self._calc_grad(model, inputs, targets)
            return self._loss, grad_dict
        else:
            return self._loss, None

    def _calc_loss(self, *inputs):
        raise NotImplementedError

    def _calc_grad(self, model, inputs, targets):
        grad_dict = OrderedDict()
        for l_name, layer in model.layers.items():
            grad_dict[l_name] = OrderedDict()
            for param_name, param in layer.params.items():
                # パラメータを微小区間動かした後の損失を計算
                model.layers[l_name].params[param_name] = param + self.eps
                moved_loss = np.mean([
                    self._calc_loss(model.forward(_input), target)
                    for _input, target in zip(inputs, targets)
                ])
                model.layers[l_name].params[param_name] = param

                # 損失Lの勾配を計算
                loss_grad = (moved_loss - self._loss) / self.eps

                grad_dict[l_name][param_name] = loss_grad

        return grad_dict


class BinaryCrossEntropy(BaseLoss):
    def __init__(self, is_sigmoid=False):
        super().__init__()
        self.is_sigmoid = is_sigmoid

    def _calc_loss(self, predict, target):
        if self.is_sigmoid:
            assert True, "Unimplemented."
        else:
            return target * np.log(1 + np.exp(-predict)) + (1 - target) * np.log(1 + np.exp(predict))
