import numpy as np
import synthetic_generators
import scipy
import torch


def expected_normal_kurtosis(data):
    _, dimensions = data.shape
    return dimensions * (dimensions + 2)


def expected_normal_entropy(covariance):
    d = covariance.shape[0]
    return 0.5 * d * (
        1 + torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)))
    ) + 0.5 * torch.logdet(covariance)


def calculate_moments(data, num_moments=4, eps=1e-8):
    """
    Calculate the specified number of moments (mean, covariance, skewness, and kurtosis) of a set of n-dimensional points.

    Args:
        points (torch.Tensor): A tensor of shape (num_points, n_dims) representing the points.
        num_moments (int): The number of moments to calculate (1 to 4).
                           1: Mean
                           2: Mean and Covariance
                           3: Mean, Covariance, and Skewness
                           4: Mean, Covariance, Skewness, and Kurtosis
        eps (float): A small value to add to the denominator for numerical stability.

    Returns:
        tuple: A tuple containing the calculated moments based on the specified num_moments.
    """
    if num_moments < 1 or num_moments > 4:
        raise ValueError("num_moments must be between 1 and 4")

    num_points = data.shape[0]
    mean = torch.mean(data, dim=0)

    if num_moments == 1:
        return (mean,)

    centered_points = data - mean
    if centered_points.dim() == 1:
        covariance = torch.var(centered_points, unbiased=False)
    elif centered_points.dim() == 2:
        covariance = torch.matmul(centered_points.T, centered_points) / (num_points - 1)
    else:
        raise ValueError("Expect data to have 1 or 2 dimensions.")

    if num_moments == 2:
        return mean, covariance

    # Calculate the inverse of the covariance matrix
    inv_covariance = torch.inverse(covariance + torch.eye(covariance.shape[0]) * eps)

    # Mahalanobis distance squared for each point
    mahalanobis_distances = torch.sqrt(
        torch.sum(
            torch.matmul(centered_points, inv_covariance) * centered_points, dim=1
        )
    )

    # Skewness and Kurtosis using Mahalanobis distance
    skewness = torch.mean(mahalanobis_distances**3) / (
        num_points * (eps + torch.mean(mahalanobis_distances**2)) ** 1.5
    )

    if num_moments == 3:
        return mean, covariance, skewness

    kurtosis = torch.mean(mahalanobis_distances**4)
    # This is a zero centered version of kurtosis.
    # kurtosis = torch.mean(mahalanobis_distances ** 4) / (num_points * (eps + torch.mean(mahalanobis_distances ** 2)) ** 2)

    return mean, covariance, skewness, kurtosis


def calculate_jarque_bera(sample_count, dimensions, skewness, kurtosis):
    """
    Perform the Jarque-Bera test on sample data to determine if the data are likely to have come from a Gaussian distribution.

    - A higher Jarque-Bera test statistic indicates greater deviation from normality.
    - A p-value less than a chosen significance level (commonly 0.05 or 0.01) suggests rejecting the null hypothesis that the data follow a normal distribution for the given dimension.
    This means that low p-values indicate significant evidence against the data being normally distributed, while high p-values suggest that the data could plausibly be from a normal distribution.


    Parameters:
    sample_count (int): The number of observations or samples in the dataset.
    skewness (torch.Tensor): A tensor containing the skewness of the dataset for each dimension.
                             This should be a single value if dealing with unidimensional data or a tensor of skewness values for multidimensional data.
    kurtosis (torch.Tensor): A tensor containing the kurtosis of the dataset for each dimension.
                             Like skewness, this should be a single value for unidimensional data or a tensor of kurtosis values for multidimensional data.

    Returns:
    float: The Jarque-Bera test statistic.
    torch.Tensor: P-value of the test (not precisely calculated here, more of an illustrative placeholder).
    """

    # Calculate the Jarque-Bera test statistic for each dimension
    expected_kurtosis = dimensions * (dimensions + 2)
    jb_stat = (sample_count / 6) * (
        skewness.pow(2) + 0.25 * (kurtosis - expected_kurtosis).pow(2)
    )

    # Two degrees of freedom: skewness and kurtosis
    p_value = 1 - scipy.stats.chi2.cdf(
        jb_stat, df=2
    )  # Assuming two degrees of freedom for the test

    # For now, returning the JB statistic without a p-value
    return jb_stat, p_value  # Placeholder for p-value


def calculate_shapiro_wilk(data):
    """
    Perform the Shapiro-Wilk test for normality.

    - The test statistic ranges from 0 to 1, where a value closer to 1 indicates that the data are more likely to follow a normal distribution.
    - The p-value helps in deciding whether to reject the null hypothesis of normality. A low p-value (typically < 0.05) suggests rejecting the null hypothesis, indicating the data do not follow a normal distribution.
    - A high p-value indicates insufficient evidence to reject the null hypothesis, suggesting that the data could be normally distributed.

    Notes
    - This test is particularly sensitive to deviations from normality and is most effective on small to moderate sample sizes (n<50).
    - For large datasets, the test's sensitivity can sometimes lead to small deviations being identified as significant, thus rejecting the null hypothesis for even trivial deviations from normality.

    Example:
    If you receive a p-value of 0.04, it suggests rejecting the null hypothesis at the 5% significance level, concluding that the data are not normally distributed.

    Parameters:
    data (numpy.ndarray): A 1-D array of sample data.

    Returns:
    float: The test statistic.
    float: The p-value of the test.
    """
    print(data.size())
    test_statistic, p_value = scipy.stats.shapiro(data)
    return test_statistic, p_value


def calculate_anderson_darling(data):
    """
     Perform the Anderson-Darling test for normality on a sample 1d dataset.

     - The test statistic is a measure of how well the data conform to a specified distribution - in this case, the normal distribution.
     - Compare the test statistic to the critical values: if the test statistic is larger than the critical value at the .05 significance level, the null hypothesis that the data come from a normal distribution is rejected.
     - Critical values are typically provided for several significance levels; if the test statistic exceeds these, it indicates stronger evidence against the null hypothesis.

     Notes:
     - This test is sensitive to deviations from normality, especially in the tails of the distribution.
     - It is applicable to both small and large datasets but is particularly recommended for more comprehensive assessments of normality where tail properties are of concern.

    Parameters:
     data (numpy.ndarray): A 1-D array of sample data, typically representing measurements or observations from a single variable.

     Returns:
     float: The Anderson-Darling test statistic, which measures the closeness of the data to a normal distribution.
     list: A list of critical values at various significance levels.
     list: A list of corresponding significance levels.

     Raises:
     ValueError: If the input data is not a 1-D array.
    """
    if data.ndim == 2 and data.size()[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        raise ValueError("Data must be a one-dimensional tensor.")
    result = scipy.stats.anderson(data.numpy(), dist="norm")
    return result


def calculate_kolmogorov_smirnov(data, mean=0, std=1):
    """
    Perform the one-sample Kolmogorov-Smirnov test for normality.

    - The test statistic is the maximum deviation between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution.
    - A test statistic close to 0 indicates that the sample distribution closely matches the theoretical distribution.
    - A test statistic close to 1 indicates a significant divergence between the sample and the theoretical distributions.
    - A small p-value (typically < 0.05) suggests rejecting the null hypothesis, indicating that the data do not follow the specified normal distribution.
    - A large p-value suggests insufficient evidence to reject the null hypothesis, meaning the data could plausibly come from the normal distribution.

    Notes:
    - This test is sensitive to differences in both the center location and shape of the distribution compared to the theoretical normal distribution.
    - For accurate results, ensure the data does not have ties (identical values), as K-S test assumptions include the continuity of the distribution.

    Parameters:
    data (numpy.ndarray): A 1-D array of sample data, typically representing measurements or observations.
    mean (float): The mean of the normal distribution to test against (default 0).
    std (float): The standard deviation of the normal distribution to test against (default 1).

    Returns:
    float: The K-S test statistic, which is the maximum difference between the observed and expected cumulative distributions.
    float: The p-value of the test, which indicates the probability of an observed (or more extreme) result assuming the null hypothesis is true.

    """
    if data.ndim == 2 and data.size()[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        raise ValueError("Data must be a one-dimensional tensor.")

    # Normalize data to standard normal if not already
    normalized_data = (data - mean) / std

    # Perform the K-S test against a standard normal distribution
    test_statistic, p_value = scipy.stats.kstest(normalized_data.numpy(), "norm")

    return test_statistic, p_value


def calculate_mardias(data):
    """
    Perform Mardia's test of skewness and kurtosis to assess multivariate normality using PyTorch.

    Parameters:
    data (torch.Tensor): A 2-D tensor where each row represents an observation and each column a variable.

    Returns:
    dict: A dictionary containing the following key-value pairs:
        - 'skewness': Mardia's multivariate skewness value.
        - 'skewness_p_value': The p-value for the skewness test.
        - 'kurtosis': Mardia's multivariate kurtosis value.
        - 'kurtosis_p_value': The p-value for the kurtosis test.
        - 'combined_p_value': The p-value for the combined test.
    """
    num_points, dimensions = data.shape
    _, _, skew, kurt = calculate_moments(data, 4)

    # Calculate the p-value for the skewness test
    degrees_freedom_skewness = dimensions * (dimensions + 1) * (dimensions + 2) // 6
    skewness_test_stat = num_points * skew / 6
    skewness_p_value = 1 - scipy.stats.chi2.cdf(
        skewness_test_stat.item(), degrees_freedom_skewness
    )

    # Calculate the p-value for the kurtosis test
    # kurtosis_test_stat = num_points * kurt / 8
    expected_kurtosis = expected_normal_kurtosis(data)
    excess_kurtosis = kurt - expected_kurtosis
    kurtosis_test_stat = excess_kurtosis / (8 * expected_kurtosis / num_points) ** 0.5
    kurtosis_p_value = 2 * (1 - scipy.stats.norm.cdf(abs(kurtosis_test_stat.item())))

    # Calculate the p-value for the combined test
    combined_test_stat = skewness_test_stat + kurtosis_test_stat**2
    df_combined = 2
    combined_p_value = 1 - scipy.stats.chi2.cdf(combined_test_stat.item(), df_combined)

    return {
        "skewness": skew.item(),
        "skewness_p_value": skewness_p_value,
        "kurtosis": kurt.item(),
        "excess_kurtosis": excess_kurtosis.item(),
        "kurtosis_p_value": kurtosis_p_value,
        "combined_p_value": combined_p_value,
    }


def calculate_roystons(data):
    """
    Perform Royston's multivariate normality test on a dataset.

    The Shapiro-Wilk test, which is a component of Royston's test for assessing multivariate
    normality, is traditionally more accurate for smaller sample sizes (typically N < 5000).
    For larger datasets, the computation of the test statistic can be influenced by the sheer
    volume of data, potentially leading to inaccuracies in estimating the p-value. This is due
    to the complexity of calculations involved and the assumptions about distribution
    characteristics at such scale.

    This test statistic is calculated by combining the test statistics (specifically the
    p-values) from individual Shapiro-Wilk tests applied to each dimension of your dataset. A
    higher test statistic generally indicates stronger evidence against the null hypothesis, but
    it's the p-value that will provide a clear decision criterion.

    Parameters:
        data (torch.Tensor): A PyTorch tensor where each column represents a variable.

    Returns:
        float, float: The combined test statistic and the p-value indicating the test result.
    """
    # Assuming data is a PyTorch tensor, convert to NumPy for Shapiro-Wilk test
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Calculate Shapiro-Wilk statistics for each variable
    sw_statistics = [scipy.stats.shapiro(data[:, i]) for i in range(data.shape[1])]

    # Extract test statistics and p-values
    p_values = np.array([stat[1] for stat in sw_statistics])

    # Combine the p-values using Fisher's method
    combined_statistic, combined_p_value = scipy.stats.combine_pvalues(
        p_values, method="fisher"
    )

    return combined_statistic, combined_p_value


def calculate_henze_zirkler(data):
    """
    Perform Henze-Zirkler's multivariate normality test using PyTorch.

    Parameters:
        data (torch.Tensor): A PyTorch tensor where each row represents an observation and each column a variable.

    Returns:
        float, float: The test statistic and the p-value indicating the test result.
    """
    n, d = data.shape
    mean = torch.mean(data, dim=0)
    cov = torch.cov(data.t())  # Transpose data to get variables as rows
    cov_inv = torch.linalg.inv(cov)

    # Calculate Mahalanobis distances
    diff = data - mean
    mahalanobis_dist = torch.diag(torch.mm(torch.mm(diff, cov_inv), diff.t()))

    # Henze-Zirkler test statistic
    b2 = (1 + (d + 1) / n) / 2
    epsilon = 0.005  # Small tolerance for b2 calculation
    if b2 > 0.5 and b2 <= 0.5 + epsilon:
        b2 = 0.5 - epsilon  # Truncate b2 to 0.5 if it slightly exceeds

    S = torch.sum(torch.exp(-b2 * mahalanobis_dist / 2))

    # Ensure the argument inside the sqrt is non-negative
    argument = 2 * n * b2**d / (1 - 2 * b2) ** (d / 2) ** 2
    if argument < 0:
        return None, None  # Adjust the method or handle error
    argument = torch.tensor(argument, dtype=torch.float32)
    test_statistic = (
        S - n * (1 - b2) ** (d / 2) / (1 - 2 * b2) ** (d / 2)
    ) / torch.sqrt(argument)

    # Compute p-value using SciPy as PyTorch does not support chi-squared CDF
    test_statistic_np = test_statistic.item()  # Convert to Python scalar
    p_value = 1 - scipy.stats.chi2.cdf(test_statistic_np, d)

    return test_statistic_np, p_value


def calculate_energy_test(
    data,
    critical_value=None,
    mean=None,
    cov=None,
    samples=None,
    simulations=None,
    alpha=0.05,
):
    """
    Calculate the Energy Test for Normality on a dataset. This test is useful in statistical
    analysis to determine if a dataset significantly deviates from a Gaussian (normal)
    distribution.

    1. Compute pairwise distances between all points in your dataset.
    2. Calculate the energy statistic using the distances, which involves the mean of the pairwise distances and the distances between sample points and a reference point.
    3. Return the test statistic. Normally, you would compare this against critical values to determine the normality, but since the focus here is on computation, we'll compute the statistic first.

    To determine normality by comparing the calculated energy statistic against critical values,
    you need a reference distribution of the energy statistic under the null hypothesis that the
    data is normally distributed. Since the distribution of the energy statistic under the null
    hypothesis typically doesn't have a simple analytical form, it's common to estimate these
    critical values through simulation (bootstrapping) or by using asymptotic approximations if
    available.

    This function can also simulate critical values if necessary parameters are provided.

    Parameters:
        data (torch.Tensor): The data points as a PyTorch tensor, shape (n_samples, n_features)

    Returns:
        float: The energy statistic
    """
    if (
        critical_value is None
        and mean is not None
        and cov is not None
        and samples is not None
        and simulations is not None
    ):
        # calculate the critical value if we have been given a reference Gaussian
        simulated_data = [
            torch.tensor(
                np.random.multivariate_normal(mean, cov, samples), dtype=torch.float32
            )
            for _ in range(simulations)
        ]
        energy_statistics = [
            calculate_energy_test(data=sim_data)["energy_statistic"]
            for sim_data in simulated_data
        ]
        critical_value = np.percentile(energy_statistics, 100 * (1 - alpha))

    # Compute pairwise distances
    dist_matrix = torch.cdist(data, data, p=2)

    # Calculate the mean pairwise distance
    n = data.size(0)
    mean_dist = torch.mean(dist_matrix)

    # Calculate distances from the mean (the centroid of the data)
    mean_vector = torch.mean(data, dim=0)
    dist_from_mean = torch.norm(data - mean_vector, dim=1, p=2)

    # Calculate the mean of these distances
    mean_dist_from_mean = torch.mean(dist_from_mean)

    # Energy statistic computation
    energy_statistic = 2 * mean_dist - mean_dist_from_mean

    # Determine if we should reject the null hypothesis
    reject_null_hypothesis = (
        energy_statistic > critical_value if critical_value is not None else None
    )

    return {
        "energy_statistic": energy_statistic.item(),
        "critical_value": critical_value,
        "reject_null_hypothesis": reject_null_hypothesis,
    }


def k_nearest_neighbors_entropy(data, k=3):
    """
    Estimate entropy using the k-nearest neighbors method for multivariate data in PyTorch.
    """
    # Calculate pairwise distances
    dist_matrix = torch.cdist(data, data, p=2)

    # Get distances to the k-nearest neighbors, excluding the point itself
    # torch.topk can be used to find the k smallest distances
    k_distances, _ = torch.topk(dist_matrix, k + 1, largest=False, sorted=True)
    radius = k_distances[:, k]  # Distance to the k-th nearest neighbor (exclude itself)

    # Calculate entropy
    entropy_estimate = (
        torch.mean(torch.log(radius)) * data.shape[1]
        + torch.log(torch.tensor(k))
        - torch.log(torch.tensor(data.shape[0] - 1))
    )
    return entropy_estimate.item()


def calculate_density_knn(data, k):
    """
    Estimate the probability density using the k-nearest neighbors.
    """
    n_samples, n_features = data.size()
    distances = torch.cdist(data, data)
    _, indices = torch.topk(distances, k + 1, dim=1, largest=False)
    radii = distances[torch.arange(n_samples), indices[:, -1]]

    volume_unit_ball = torch.pow(
        torch.tensor(torch.pi), n_features / 2
    ) / scipy.special.gamma(n_features / 2 + 1)
    volumes = volume_unit_ball * torch.pow(radii, n_features)

    density = k / (n_samples * volumes)
    return density


def calculate_negentropy_knn(data, k=3):
    """
    Calculate the negentropy of a dataset to measure the non-Gaussianity.

    Negentropy is computed as the difference between the entropy of a Gaussian distribution
    with the same covariance as the data and the estimated entropy of the data.

    Negentropy is always non-negative and zero if and only if the data follows
    a Gaussian distribution. It is a measure of deviation from the Gaussianity,
    where larger values indicate greater deviation.

    Parameters:
        data (torch.Tensor): A PyTorch tensor containing the data set for which negentropy should be calculated.

    Returns:
        float: The calculated negentropy of the data.
    """
    density = calculate_density_knn(data, k)
    data_entropy = -torch.mean(torch.log(density))
    gaussian_entropy_val = expected_normal_entropy(torch.cov(data.T))
    negentropy = gaussian_entropy_val - data_entropy
    return negentropy.item()


def calculate_negentropy_kde(data):
    """
    Compute the negentropy using Kernel Density Estimation (KDE).
    """
    _, n_features = data.size()

    np_data = data.T.numpy()
    # Compute the Gaussian KDE
    kde = scipy.stats.gaussian_kde(np_data)

    # Estimate the probability density using KDE
    density = torch.tensor(kde(np_data), dtype=torch.float32)

    # Compute the entropy of the estimated probability density
    data_entropy = -torch.mean(torch.log(density))

    # Compute the entropy of a Gaussian distribution with the same mean and variance as the data
    covariance = torch.cov(data.T)
    gaussian_entropy = 0.5 * n_features * (
        1 + torch.log(2 * torch.tensor(torch.pi))
    ) + 0.5 * torch.log(torch.det(covariance))

    # Compute the negentropy
    negentropy = gaussian_entropy - data_entropy

    return negentropy.item()


def calculate_kld_kde(data, integration_method="mc", n_integration_samples=10000):
    """
    Compute the mutual information between features using Kernel Density Estimation (KDE).
    """
    data_np = data.cpu().numpy()  # Convert PyTorch tensor to NumPy array
    n_samples, n_features = data_np.shape

    # Precompute bounds for each feature
    lower_bounds = [np.min(data_np[:, i]) for i in range(n_features)]
    upper_bounds = [np.max(data_np[:, i]) for i in range(n_features)]

    # Compute the marginal and joint probability densities using KDE
    marginal_densities = []
    for i in range(n_features):
        feature_data = data_np[:, i]
        kde = scipy.stats.gaussian_kde(feature_data.T)
        marginal_densities.append(kde)

    joint_density = scipy.stats.gaussian_kde(data_np.T)

    # Define the integrand function for mutual information calculation
    def integrand(*args):
        marginals = [
            marginal_densities[i](np.array([arg])) for i, arg in enumerate(args)
        ]
        joint = joint_density(args)
        return joint * np.log(joint / np.prod(marginals))

    bounds = list(zip(lower_bounds, upper_bounds))

    if integration_method == "mc":
        # n_samples needs to grow exponentially with dimension count
        samples = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(n_integration_samples, n_features),
        )
        integral = np.mean([integrand(*sample) for sample in samples])
        mutual_info = integral * np.prod(
            [high - low for high, low in zip(upper_bounds, lower_bounds)]
        )
    elif integration_method == "quad":
        # nquad is slower but more accurate
        mutual_info, _ = scipy.integrate.nquad(integrand, bounds)
    else:
        raise ValueError(f"Unknown integration_method {integration_method}")

    return mutual_info


def calculate_kld_pca(data, igen_power=0.95):
    # Calculate mean and covariance of data
    mean, covariance = calculate_moments(data, 2)
    centered_data = data - mean

    # make sure centered_data is a matrix
    if centered_data.dim() == 1:
        centered_data = centered_data.unsqueeze(1)

    # Perform PCA on the conditioned covariance
    if centered_data.size(1) > 1:
        # Multivariate
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        # Sort the principal components from largest eigenvalue to smallest
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        pass
    else:
        # Univariate
        covariance = torch.var(centered_data, unbiased=True)
        sorted_eigenvalues = covariance.unsqueeze(0)  # Create a one-element tensor
        sorted_eigenvectors = torch.tensor([[1.0]])  # Eigenvector is 1 in 1D

    # Set any negative eigenvalues to zero
    sorted_eigenvalues = torch.clamp(sorted_eigenvalues, min=0)

    # Calculate the total sum of eigenvalues
    total_eigenvalue_sum = torch.sum(sorted_eigenvalues)

    # Initialize variables for iteration
    processed_eigenvalue_sum = 0
    mutual_info_sum = 0

    # Iterate over eigenvectors from largest eigenvalue to smallest
    for eigenvalue, eigenvector in zip(sorted_eigenvalues, sorted_eigenvectors.T):
        # Project the data onto the eigenvector
        projected_data = centered_data.mm(eigenvector.unsqueeze(1)).squeeze(1)

        # Calculate the 1D mutual information using KDE
        # Here, you need to ensure that 'calculate_kld_kde' is implemented in PyTorch
        mutual_info_1d = calculate_kld_kde(
            projected_data.unsqueeze(1), integration_method="quad"
        )
        mutual_info_sum += mutual_info_1d

        # Update the processed eigenvalue sum
        processed_eigenvalue_sum += eigenvalue

        # Check if the processed eigenvalue sum exceeds the specified power threshold
        if processed_eigenvalue_sum / total_eigenvalue_sum > igen_power:
            break

    return mutual_info_sum


def build_univar_gaussian_hist(bins, uniform=False):
    normal_dist = torch.distributions.normal.Normal(
        torch.tensor(0.0), torch.tensor(1.0)
    )

    if uniform:
        # Generate bin edges with equal probability mass using inverse CDF (ppf)
        bin_edges = normal_dist.icdf(torch.linspace(0, 1, bins + 1))
    else:
        # Generate equally spaced bin edges
        bin_edges = torch.linspace(-5, 5, bins + 1)

    # Calculate the probability mass for each bin using the CDF
    bin_probs = normal_dist.cdf(bin_edges[1:]) - normal_dist.cdf(bin_edges[:-1])

    # Normalize the probabilities to ensure they sum to 1 (for numerical stability)
    bin_probs = bin_probs / torch.sum(bin_probs)

    return bin_probs, bin_edges


def build_multivar_gaussian_hist(bins, dimensionality, uniform=False):
    if uniform:
        # Using SciPy's ppf function to get quantiles for equal mass bins
        bin_edges = (
            scipy.stats.chi2.ppf(np.linspace(0, 1, bins + 1), df=dimensionality) ** 0.5
        )
    else:
        # Equally spaced bin edges after determining a reasonable maximum value
        max_value = scipy.stats.chi2.ppf(0.99, df=dimensionality) ** 0.5
        bin_edges = np.linspace(0, max_value, bins + 1)

    # Calculate probability mass for each bin using CDF
    bin_probs = scipy.stats.chi2.cdf(
        bin_edges[1:] ** 2, df=dimensionality
    ) - scipy.stats.chi2.cdf(bin_edges[:-1] ** 2, df=dimensionality)

    # Normalize probabilities (this should already sum to 1, but good to ensure numerical stability)
    bin_probs = bin_probs / np.sum(bin_probs)

    # Convert results to PyTorch tensors
    bin_probs_torch = torch.from_numpy(bin_probs.astype(np.float32))
    bin_edges_torch = torch.from_numpy(bin_edges.astype(np.float32))

    return bin_probs_torch, bin_edges_torch


def build_univar_hist(data, bin_edges):
    # Initialize histogram tensor
    hist = torch.zeros(bin_edges.shape[0] - 1, dtype=torch.float32)

    # Calculate the index for each data point
    indices = torch.bucketize(data, bin_edges, right=True)
    indices = torch.clamp(indices - 1, 0, hist.shape[0] - 1)

    # Count occurrences in each bin
    for i in range(hist.shape[0]):
        hist[i] = torch.sum(indices == i)

    epsilon = 1e-10
    hist = torch.clamp(hist, min=epsilon)

    # Normalize the histogram to obtain probabilities
    hist = hist / torch.sum(hist)

    return hist


def kld_for_hist(hist1, hist2):
    # print(f"hist1 {hist1}")
    # print(f"hist2 {hist2}")
    # Calculate the KLD for each bin
    kld_bins = hist1 * torch.log(hist1 / hist2)
    kld_sum = torch.sum(kld_bins)
    # print(f"kld_bins {kld_bins}")

    return kld_sum


def calculate_kld_univar_hist(data, bins, mean=None, covariance=None, uniform=False):
    if data.dim() == 1:
        data = data.unsqueeze(1)

    if mean is None:
        mean = torch.mean(data, dim=0)
    if covariance is None:
        covariance = torch.var(data, unbiased=False)

    centered_data = data - mean

    # Build the standard normal histogram
    reference_hist, bin_edges = build_univar_gaussian_hist(bins, uniform)

    # Calculate the Mahalanobis distances
    signed_distances = centered_data / torch.sqrt(covariance)

    # Build the histogram of the Mahalanobis distances using custom histogram function
    data_hist = build_univar_hist(signed_distances, bin_edges)

    # Calculate the KLD between the data histogram and the reference histogram
    return kld_for_hist(data_hist, reference_hist)


def calculate_kld_multivar_hist(
    data, bins, mean=None, cov=None, inv_cov=None, uniform=False
):
    n_samples, n_features = data.size()

    if mean is None:
        mean = torch.mean(data, dim=0)

    centered_points = data - mean
    if cov is None:
        cov = torch.matmul(centered_points.T, centered_points) / (n_samples - 1)

    if inv_cov is None:
        inv_cov = torch.inverse(cov + 1e-8 * torch.eye(cov.size(0)))

    # Assume build_multivar_gaussian_hist is implemented correctly in PyTorch
    reference_hist, bin_edges = build_multivar_gaussian_hist(bins, n_features, uniform)

    # Calculate the Mahalanobis distances
    centered_data = data - mean
    mahalanobis_distances_squared = torch.sum(
        (centered_data @ inv_cov) * centered_data, axis=1
    )
    mahalanobis_distances = torch.sqrt(mahalanobis_distances_squared)

    # Build the histogram of the Mahalanobis distances using a custom function for variable bin widths
    data_hist = build_univar_hist(mahalanobis_distances, bin_edges)

    # Normalize the histogram to obtain probabilities
    data_hist = data_hist / torch.sum(data_hist)

    # Assume kld_for_histograms is implemented correctly in PyTorch
    kld = kld_for_hist(data_hist, reference_hist)

    return kld


if __name__ == "__main__":

    def main():
        # Example data
        num_points = 10000
        dimensions = 3
        data = synthetic_generators.sample_gaussian(num_points, dimensions)

        moments = calculate_moments(data, 4)
        print("Mean:\n", moments[0])
        print("Covariance:\n", moments[1])
        print("Skewness:\n", moments[2])
        print("Kurtosis:\n", moments[3])
        print(
            f"Jarque-Bera {calculate_jarque_bera(num_points, dimensions, moments[2], moments[3])}"
        )
        print(f"Shapiro-Wilk {calculate_shapiro_wilk(data[:1000])}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 0])}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 1])}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 2])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 0])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 1])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 2])}")
        print(f"Mardia's Test {calculate_mardias(data)}")
        print(f"Royston's Test {calculate_roystons(data[0:1000])}")
        print(f"Henze-Zirkler Test {calculate_henze_zirkler(data)}")
        print(f"Energy Test {calculate_energy_test(data, critical_value=3.0)}")
        print(
            f"Energy Test {calculate_energy_test(data, mean=torch.zeros(dimensions), cov=torch.eye(dimensions), samples=1000, simulations=100)}"
        )
        print(f"Negentropy k-NN  3 {calculate_negentropy_knn(data, k=3)}")
        print(f"Negentropy k-NN  6 {calculate_negentropy_knn(data, k=6)}")
        print(f"Negentropy KDE {calculate_negentropy_kde(data)}")
        print(
            f"Kullback-Leibler Divergence x KDE {calculate_kld_kde(data[:, 0].unsqueeze(1), integration_method = 'quad')}"
        )
        print(
            f"Kullback-Leibler Divergence y KDE {calculate_kld_kde(data[:, 1].unsqueeze(1), integration_method = 'quad')}"
        )
        print(
            f"Kullback-Leibler Divergence z KDE {calculate_kld_kde(data[:, 2].unsqueeze(1), integration_method = 'quad')}"
        )
        print(f"Kullback-Leibler Divergence KDE {calculate_kld_kde(data)}")
        print(f"Kullback-Leibler Divergence PCA x {calculate_kld_pca(data[:, 0])}")
        print(f"Kullback-Leibler Divergence PCA xy {calculate_kld_pca(data[:, 0:1])}")
        print(f"Kullback-Leibler Divergence PCA xyz {calculate_kld_pca(data)}")
        print(
            f"Kullback-Leibler Divergence Hist x {calculate_kld_univar_hist(data[:, 0], 10)}"
        )
        print(
            f"Kullback-Leibler Divergence Hist y {calculate_kld_univar_hist(data[:, 1], 10)}"
        )
        print(
            f"Kullback-Leibler Divergence Hist z {calculate_kld_univar_hist(data[:, 2], 10)}"
        )
        print(f"KLD Hist z u {calculate_kld_univar_hist(data[:, 2], 10, uniform=True)}")
        print(f"KLD Hist {calculate_kld_multivar_hist(data, 10)}")
        print(f"KLD Hist u {calculate_kld_multivar_hist(data, 10, uniform=True)}")

    main()
