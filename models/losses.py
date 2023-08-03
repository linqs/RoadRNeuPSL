import torch


def binary_cross_entropy(outputs, truth, indices):
    """
    Computes the binary cross entropy loss for the given outputs and targets.
    The targets are aligned with the outputs using the given indices.
    :param outputs: the outputs of the model. dict of tensors of shape [batch_size, num_queries, num_classes]
    :param truth: the ground truth labels. tensor of shape [batch_size, num_queries, num_classes]
    :param indices: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
    :return: the computed binary cross entropy loss
    """

    # Align the outputs and targets using the given indices and flatten.
    aligned_outputs = torch.stack([outputs[i, indices[i, 0, :], :] for i in range(outputs.shape[0])]).flatten(0, 1)
    aligned_truth = torch.stack([truth[i, indices[i, 1, :], :] for i in range(truth.shape[0])]).flatten(0, 1)

    return torch.nn.functional.binary_cross_entropy(aligned_outputs, aligned_truth)

