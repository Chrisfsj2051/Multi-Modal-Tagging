"""Provides functions to help with evaluating models."""

import numpy as np

import src.utils.mmt.utils.metrics.average_precision_calculator as ap_calculator
import src.utils.mmt.utils.metrics.mean_average_precision_calculator as map_calculator
from src.utils.mmt.utils import PRCalculator
from src.utils.mmt.utils import PRCalculatorPerTag


###
###Training utils
###
def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

    Args:
      feat_vector: the input 1-d vector.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.
    Returns:
      A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def flatten(l):
    """Merges a list of lists into a single list."""
    return [item for sublist in l for item in sublist]


def get_tag_stat(labels):
    """get freq num of each tag."""
    num_classes = labels.shape[1]
    num_stat = np.zeros(num_classes)
    for i in range(num_classes):
        num_stat[i] = np.sum(labels[:, i])
    return num_stat


def get_tag_correlation(preds, labels, top_k=10):
    n_example, n_class = preds.shape
    tag_correlation = np.zeros((n_class, n_class))
    top_k = min(n_class, top_k)
    # convert pred to top_k index
    pred_indx = np.zeros((n_example, n_class), dtype=np.int8)
    for i in range(n_example):
        for idx in np.argpartition(preds[i], -top_k)[-top_k:]:
            pred_indx[i][idx] = 1
    # get correlation matrix
    for i in range(n_example):
        label_index = np.nonzero(labels[i])[0]
        pred_index = np.nonzero(pred_indx[i])[0]
        for li in label_index:
            for pi in pred_index:
                tag_correlation[li][pi] += 1
    return tag_correlation


def get_tag_confidence(predictions, labels):
    n_example, n_class = predictions.shape
    tag_confidence = np.zeros(n_class)
    for i in range(n_example):
        label_index = np.nonzero(labels[i])[0]
        for j in label_index:
            tag_confidence[j] += predictions[i][j]
    return tag_confidence


def calculate_hit_at_one(predictions, actuals):
    """Performs a local (numpy) calculation of the hit at one.

    Args:
      predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.

    Returns:
      float: The average hit at one across the entire batch.
    """
    top_prediction = np.argmax(predictions, 1)
    hits = actuals[np.arange(actuals.shape[0]), top_prediction]
    return np.average(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
    """Performs a local (numpy) calculation of the PERR.

    Args:
      predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.

    Returns:
      float: The average precision at equal recall rate across the entire batch.
    """
    aggregated_precision = 0.0
    num_videos = actuals.shape[0]
    for row in np.arange(num_videos):
        num_labels = int(np.sum(actuals[row]))
        top_indices = np.argpartition(predictions[row],
                                      -num_labels)[-num_labels:]
        item_precision = 0.0
        for label_index in top_indices:
            if predictions[row][label_index] > 0:
                item_precision += actuals[row][label_index]
        item_precision /= top_indices.size
        aggregated_precision += item_precision
    aggregated_precision /= num_videos
    return aggregated_precision


def calculate_gap(predictions, actuals, top_k=20):
    """Performs a local (numpy) calculation of the global average precision.

    Only the top_k predictions are taken for each of the videos.

    Args:
      predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      top_k: How many predictions to use per frame.

    Returns:
      float: The global average precision.
    """
    gap_calculator = ap_calculator.AveragePrecisionCalculator()
    sparse_predictions, sparse_labels, num_positives = top_k_by_class(
        predictions, actuals, top_k)
    gap_calculator.accumulate(flatten(sparse_predictions),
                              flatten(sparse_labels), sum(num_positives))
    return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
    """Extracts the top k predictions for each frame, sorted by class.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      k: the top k non-zero entries to preserve in each prediction.

    Returns:
      A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
      are lists of lists of floats. 'true_positives' is a list of scalars. The
      length of the lists are equal to the number of classes. The entries in the
      predictions variable are probability predictions, and
      the corresponding entries in the labels variable are the ground truth for
      those predictions. The entries in 'true_positives' are the number of true
      positives for each class in the ground truth.

    Raises:
      ValueError: An error occurred when the k is not a positive integer.
    """
    if k <= 0:
        raise ValueError('k must be a positive integer.')
    k = min(k, predictions.shape[1])
    num_classes = predictions.shape[1]
    prediction_triplets = []
    for video_index in range(predictions.shape[0]):
        prediction_triplets.extend(
            top_k_triplets(predictions[video_index], labels[video_index], k))
    out_predictions = [[] for v in range(num_classes)]
    out_labels = [[] for v in range(num_classes)]
    for triplet in prediction_triplets:
        out_predictions[triplet[0]].append(triplet[1])
        out_labels[triplet[0]].append(triplet[2])
    out_true_positives = [np.sum(labels[:, i]) for i in range(num_classes)]

    return out_predictions, out_labels, out_true_positives


def top_k_triplets(predictions, labels, k=20):
    """Get the top_k for a 1-d numpy array.

    Returns a sparse list of tuples in (prediction, class) format
    """
    m = len(predictions)
    k = min(k, m)
    indices = np.argpartition(predictions, -k)[-k:]
    return [(index, predictions[index], labels[index]) for index in indices]


class EvaluationMetrics(object):
    """A class to store the evaluation metrics."""
    def __init__(self, num_class, top_k, accumulate_per_tag=False):
        """Construct an EvaluationMetrics object to store the evaluation
        metrics.

        Args:
          num_class: A positive integer specifying the number of classes.
          top_k: A positive integer specifying how many predictions are considered per frame.

        Raises:
          ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
            not be constructed.
        """
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.sum_loss = 0.0
        self.map_calculator = map_calculator.MeanAveragePrecisionCalculator(
            num_class)
        self.global_ap_calculator = ap_calculator.AveragePrecisionCalculator()
        self.pr_calculator = PRCalculator()
        self.pr_calculator_per_tag = PRCalculatorPerTag(num_class)
        self.accumulate_per_tag = accumulate_per_tag

        self.top_k = top_k
        self.num_examples = 0
        self.nums_per_tag = np.zeros(num_class)
        self.tag_corrlation = np.zeros((num_class, num_class))
        self.tag_confidence = np.zeros(num_class)

    def accumulate(self, predictions, labels, loss):
        """Accumulate the metrics calculated locally for this mini-batch.

        Args:
          predictions: A numpy matrix containing the outputs of the model.
            Dimensions are 'batch' x 'num_classes'.
          labels: A numpy matrix containing the ground truth labels.
            Dimensions are 'batch' x 'num_classes'.
          loss: A numpy array containing the loss for each sample.

        Returns:
          dictionary: A dictionary storing the metrics for the mini-batch.

        Raises:
          ValueError: An error occurred when the shape of predictions and actuals
            does not match.
        """
        batch_size = labels.shape[0]
        mean_hit_at_one = calculate_hit_at_one(predictions, labels)
        mean_perr = calculate_precision_at_equal_recall_rate(
            predictions, labels)
        mean_loss = loss
        self.nums_per_tag = self.nums_per_tag + get_tag_stat(labels)
        self.tag_correlation = self.tag_correlation + get_tag_correlation(
            predictions, labels, self.top_k)
        self.tag_confidence = self.tag_confidence + get_tag_confidence(
            predictions, labels)

        self.pr_calculator.accumulate(predictions, labels)
        if self.accumulate_per_tag:
            self.pr_calculator_per_tag.accumulate(predictions, labels)

        # Take the top 20 predictions.
        sparse_predictions, sparse_labels, num_positives = top_k_by_class(
            predictions, labels, self.top_k)
        self.map_calculator.accumulate(sparse_predictions, sparse_labels,
                                       num_positives)
        self.global_ap_calculator.accumulate(flatten(sparse_predictions),
                                             flatten(sparse_labels),
                                             sum(num_positives))

        self.num_examples += batch_size
        self.sum_hit_at_one += mean_hit_at_one * batch_size
        self.sum_perr += mean_perr * batch_size
        self.sum_loss += mean_loss * batch_size

        return {
            'hit_at_one': mean_hit_at_one,
            'perr': mean_perr,
            'loss': mean_loss
        }

    def get(self):
        """Calculate the evaluation metrics for the whole epoch.

        Raises:
          ValueError: If no examples were accumulated.

        Returns:
          dictionary: a dictionary storing the evaluation metrics for the epoch. The
            dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
            aps (default nan).
        """
        if self.num_examples <= 0:
            raise ValueError('total_sample must be positive.')
        avg_hit_at_one = self.sum_hit_at_one / self.num_examples
        avg_perr = self.sum_perr / self.num_examples
        avg_loss = self.sum_loss / self.num_examples

        aps = self.map_calculator.peek_map_at_n()
        gap = self.global_ap_calculator.peek_ap_at_n()
        tag_confidence = self.tag_confidence / (self.nums_per_tag + 1e-10)

        precision_at_1 = self.pr_calculator.get_precision_at_conf(0.1)
        recall_at_1 = self.pr_calculator.get_recall_at_conf(0.1)
        precision_at_5 = self.pr_calculator.get_precision_at_conf(0.5)
        recall_at_5 = self.pr_calculator.get_recall_at_conf(0.5)

        tag_precision = self.pr_calculator_per_tag.get_precision_list(
            0.5) if self.accumulate_per_tag else []
        tag_recall = self.pr_calculator_per_tag.get_recall_list(
            0.5) if self.accumulate_per_tag else []

        epoch_info_dict = {
            'avg_hit_at_one': avg_hit_at_one,
            'avg_perr': avg_perr,
            'avg_loss': avg_loss,
            'aps': aps,
            'gap': gap,
            'num': self.nums_per_tag,
            'tag_correlation': self.tag_correlation,
            'tag_confidence': tag_confidence,
            'precision_at_1': precision_at_1,
            'recall_at_1': recall_at_1,
            'precision_at_5': precision_at_5,
            'recall_at_5': recall_at_5,
            'tag_precision': tag_precision,
            'tag_recall': tag_recall
        }
        return epoch_info_dict

    def clear(self):
        """Clear the evaluation metrics and reset the EvaluationMetrics
        object."""
        self.sum_hit_at_one = 0.0
        self.sum_perr = 0.0
        self.sum_loss = 0.0
        self.map_calculator.clear()
        self.global_ap_calculator.clear()
        self.pr_calculator.clear()
        self.pr_calculator_per_tag.clear()
        self.num_examples = 0
        self.tag_correlation = 0.0
        self.nums_per_tag = 0.0
        self.tag_confidence = 0.0
