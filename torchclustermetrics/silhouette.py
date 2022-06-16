from itertools import combinations
import torch

# Code developed in NumPy by Alexandre Abraham:
# https://gist.github.com/AlexandreAbraham/5544803  Avatar


class silhouette():

    @staticmethod
    def score(X, labels, loss=False):
        """Compute the mean Silhouette Coefficient of all samples.
        The Silhouette Coefficient is calculated using the mean intra-cluster
        distance (a) and the mean nearest-cluster distance (b) for each sample.
        The Silhouette Coefficient for a sample is ``(b - a) / max(a, b)``.
        To clarrify, b is the distance between a sample and the nearest cluster
        that b is not a part of.
        This function returns the mean Silhoeutte Coefficient over all samples.
        The best value is 1 and the worst value is -1. Values near 0 indicate
        overlapping clusters. Negative values generally indicate that a sample has
        been assigned to the wrong cluster, as a different cluster is more similar.

	Code developed in NumPy by Alexandre Abraham:
	https://gist.github.com/AlexandreAbraham/5544803  Avatar
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
                 label values for each sample
        loss : Boolean
                If True, will return negative silhouette score as 
                torch tensor without moving it to the CPU. Can therefore 
                be used to calculate the gradient using autograd.
                If False positive silhouette score as float 
                on CPU will be returned.
        Returns
        -------
        silhouette : float
            Mean Silhouette Coefficient for all samples.
        References
        ----------
        Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
            Interpretation and Validation of Cluster Analysis". Computational
            and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
        http://en.wikipedia.org/wiki/Silhouette_(clustering)
        """

        if type(labels) != type(torch.HalfTensor()):
            labels = torch.HalfTensor(labels)
        if not labels.is_cuda:
            labels = labels.cuda()

        if type(X) != type(torch.HalfTensor()):
            X = torch.HalfTensor(X)
        if not X.is_cuda:
            X = X.cuda()

        unique_labels = torch.unique(labels)

        A = _intra_cluster_distances_block(X, labels, unique_labels)
        B = _nearest_cluster_distance_block(X, labels, unique_labels)
        sil_samples = (B - A) / torch.maximum(A, B)

        # nan values are for clusters of size 1, and should be 0
        mean_sil_score = torch.mean(torch.nan_to_num(sil_samples))
        if loss:
            return - mean_sil_score
        else:
            return float(mean_sil_score.cpu().numpy())

    @staticmethod
    def _intra_cluster_distances_block(X, labels, unique_labels):
        """Calculate the mean intra-cluster distance.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        Returns
        -------
        a : array [n_samples_a]
            Mean intra-cluster distance
        """
        intra_dist = torch.zeros(labels.size(), dtype=torch.float32,
                                 device=torch.device("cuda"))
        values = [_intra_cluster_distances_block_(
                    X[torch.where(labels == label)[0]])
                    for label in unique_labels]
        for label, values_ in zip(unique_labels, values):
            intra_dist[torch.where(labels == label)[0]] = values_
        return intra_dist

    @staticmethod
    def _intra_cluster_distances_block_(subX):
        distances = torch.cdist(subX, subX)
        return distances.sum(axis=1) / (distances.shape[0] - 1)

    @staticmethod
    def _nearest_cluster_distance_block(X, labels, unique_labels):
        """Calculate the mean nearest-cluster distance for sample i.
        Parameters
        ----------
        X : array [n_samples_a, n_features]
            Feature array.
        labels : array, shape = [n_samples]
            label values for each sample
        X : array [n_samples_a, n_features]
            Feature array.
        Returns
        -------
        b : float
            Mean nearest-cluster distance for sample i
        """
        inter_dist = torch.full(labels.size(), torch.inf,
                                 dtype=torch.float32,
                                device=torch.device("cuda"))
        # Compute cluster distance between pairs of clusters

        label_combinations = torch.combinations(unique_labels, 2)

        values = [_nearest_cluster_distance_block_(
                    X[torch.where(labels == label_a)[0]],
                    X[torch.where(labels == label_b)[0]])
                    for label_a, label_b in label_combinations]

        for (label_a, label_b), (values_a, values_b) in \
                zip(label_combinations, values):

                indices_a = torch.where(labels == label_a)[0]
                inter_dist[indices_a] = torch.minimum(values_a, inter_dist[indices_a])
                del indices_a
                indices_b = torch.where(labels == label_b)[0]
                inter_dist[indices_b] = torch.minimum(values_b, inter_dist[indices_b])
                del indices_b
        return inter_dist

    @staticmethod
    def _nearest_cluster_distance_block_(subX_a, subX_b):
        dist = torch.cdist(subX_a, subX_b)
        dist_a = dist.mean(axis=1)
        dist_b = dist.mean(axis=0)
        return dist_a, dist_b