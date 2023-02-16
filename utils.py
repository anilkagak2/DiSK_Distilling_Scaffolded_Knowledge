import torch
import torch.nn as nn

import numpy as np
from scipy.special import softmax

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = self.acc_matrix[:,index]


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop,self.bin_score)

        return sce/self.n_class

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)





def count_parameters_in_MB(model):
    return count_parameters(model, "mb")


def count_parameters(model_or_parameters, unit="mb"):
    if isinstance(model_or_parameters, nn.Module):
        counts = sum(np.prod(v.size()) for v in model_or_parameters.parameters())
    elif isinstance(models_or_parameters, nn.Parameter):
        counts = models_or_parameters.numel()
    elif isinstance(models_or_parameters, (list, tuple)):
        counts = sum(count_parameters(x, None) for x in models_or_parameters)
    else:
        counts = sum(np.prod(v.size()) for v in model_or_parameters)
    if unit.lower() == "kb" or unit.lower() == "k":
        counts /= 2 ** 10  # changed from 1e3 to 2^10
    elif unit.lower() == "mb" or unit.lower() == "m":
        counts /= 2 ** 20  # changed from 1e6 to 2^20
    elif unit.lower() == "gb" or unit.lower() == "g":
        counts /= 2 ** 30  # changed from 1e9 to 2^30
    elif unit is not None:
        raise ValueError("Unknow unit: {:}".format(unit))
    return counts


def get_model_infos(model, shape):
    # model = copy.deepcopy( model )

    model = add_flops_counting_methods(model)
    # model = model.cuda()
    model.eval()

    # cache_inputs = torch.zeros(*shape).cuda()
    # cache_inputs = torch.zeros(*shape)
    cache_inputs = torch.rand(*shape)
    if next(model.parameters()).is_cuda:
        cache_inputs = cache_inputs.cuda()
    # print_log('In the calculating function : cache input size : {:}'.format(cache_inputs.size()), log)
    with torch.no_grad():
        _____ = model(cache_inputs)
    FLOPs = compute_average_flops_cost(model) / 1e6
    Param = count_parameters_in_MB(model)

    if hasattr(model, "auxiliary_param"):
        aux_params = count_parameters_in_MB(model.auxiliary_param())
        print("The auxiliary params of this model is : {:}".format(aux_params))
        print(
            "We remove the auxiliary params from the total params ({:}) when counting".format(
                Param
            )
        )
        Param = Param - aux_params

    # print_log('FLOPs : {:} MB'.format(FLOPs), log)
    torch.cuda.empty_cache()
    model.apply(remove_hook_function)
    return FLOPs, Param


# ---- Public functions
def add_flops_counting_methods(model):
    model.__batch_counter__ = 0
    add_batch_counter_hook_function(model)
    model.apply(add_flops_counter_variable_or_reset)
    model.apply(add_flops_counter_hook_function)
    return model


def compute_average_flops_cost(model):
    """
    A method that will be available after add_flops_counting_methods() is called on a desired net object.
    Returns current mean flops consumption per image.
    """
    batches_count = model.__batch_counter__
    flops_sum = 0
    # or isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.MaxPool2d) \
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.Conv1d)
            or hasattr(module, "calculate_flop_self")
        ):
            flops_sum += module.__flops__
    return flops_sum / batches_count


# ---- Internal functions
def pool_flops_counter_hook(pool_module, inputs, output):
    batch_size = inputs[0].size(0)
    kernel_size = pool_module.kernel_size
    out_C, output_height, output_width = output.shape[1:]
    assert out_C == inputs[0].size(1), "{:} vs. {:}".format(out_C, inputs[0].size())

    overall_flops = (
        batch_size * out_C * output_height * output_width * kernel_size * kernel_size
    )
    pool_module.__flops__ += overall_flops


def self_calculate_flops_counter_hook(self_module, inputs, output):
    overall_flops = self_module.calculate_flop_self(inputs[0].shape, output.shape)
    self_module.__flops__ += overall_flops


def fc_flops_counter_hook(fc_module, inputs, output):
    batch_size = inputs[0].size(0)
    xin, xout = fc_module.in_features, fc_module.out_features
    assert xin == inputs[0].size(1) and xout == output.size(1), "IO=({:}, {:})".format(
        xin, xout
    )
    overall_flops = batch_size * xin * xout
    if fc_module.bias is not None:
        overall_flops += batch_size * xout
    fc_module.__flops__ += overall_flops


def conv1d_flops_counter_hook(conv_module, inputs, outputs):
    batch_size = inputs[0].size(0)
    outL = outputs.shape[-1]
    [kernel] = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    conv_per_position_flops = kernel * in_channels * out_channels / groups

    active_elements_count = batch_size * outL
    overall_flops = conv_per_position_flops * active_elements_count

    if conv_module.bias is not None:
        overall_flops += out_channels * active_elements_count
    conv_module.__flops__ += overall_flops


def conv2d_flops_counter_hook(conv_module, inputs, output):
    batch_size = inputs[0].size(0)
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    conv_per_position_flops = (
        kernel_height * kernel_width * in_channels * out_channels / groups
    )

    active_elements_count = batch_size * output_height * output_width
    overall_flops = conv_per_position_flops * active_elements_count

    if conv_module.bias is not None:
        overall_flops += out_channels * active_elements_count
    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, inputs, output):
    # Can have multiple inputs, getting the first one
    inputs = inputs[0]
    batch_size = inputs.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_hook_function(module):
    if not hasattr(module, "__batch_counter_handle__"):
        handle = module.register_forward_hook(batch_counter_hook)
        module.__batch_counter_handle__ = handle


def add_flops_counter_variable_or_reset(module):
    if (
        isinstance(module, torch.nn.Conv2d)
        or isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv1d)
        or isinstance(module, torch.nn.AvgPool2d)
        or isinstance(module, torch.nn.MaxPool2d)
        or hasattr(module, "calculate_flop_self")
    ):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if isinstance(module, torch.nn.Conv2d):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(conv2d_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.Conv1d):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(conv1d_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.Linear):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(fc_flops_counter_hook)
            module.__flops_handle__ = handle
    elif isinstance(module, torch.nn.AvgPool2d) or isinstance(
        module, torch.nn.MaxPool2d
    ):
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(pool_flops_counter_hook)
            module.__flops_handle__ = handle
    elif hasattr(module, "calculate_flop_self"):  # self-defined module
        if not hasattr(module, "__flops_handle__"):
            handle = module.register_forward_hook(self_calculate_flops_counter_hook)
            module.__flops_handle__ = handle


def remove_hook_function(module):
    hookers = ["__batch_counter_handle__", "__flops_handle__"]
    for hooker in hookers:
        if hasattr(module, hooker):
            handle = getattr(module, hooker)
            handle.remove()
    keys = ["__flops__", "__batch_counter__", "__flops__"] + hookers
    for ckey in keys:
        if hasattr(module, ckey):
            delattr(module, ckey)
