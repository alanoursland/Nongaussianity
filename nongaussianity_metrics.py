import numpy as np
import synthetic_generators
import scipy
import torch


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
    covariance = torch.matmul(centered_points.T, centered_points) / (num_points - 1)

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


def expected_normal_kurtosis(data):
    _, dimensions = data.shape
    return dimensions * (dimensions + 2)


def calculate_jarque_bera(sample_count, skewness, kurtosis):
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
    jb_stat = (sample_count / 6) * (skewness.pow(2) + 0.25 * (kurtosis - 3).pow(2))

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


if __name__ == "__main__":

    def main():
        # Example data
        num_points = 10000
        dimensions = 3
        data = synthetic_generators.sample_gaussian(num_points, dimensions)

        # Calculate up to the 4th moment
        moments = calculate_moments(data, 4)

        print("Mean:\n", moments[0])
        print("Covariance:\n", moments[1])
        print("Skewness:\n", moments[2])
        print("Kurtosis:\n", moments[3])

        print(
            f"Jarque-Bera {calculate_jarque_bera(num_points, moments[2], moments[3])}"
        )
        print(f"Shapiro-Wilk {calculate_shapiro_wilk(data)}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 0])}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 1])}")
        print(f"Anderson-Darling {calculate_anderson_darling(data[:, 2])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 0])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 1])}")
        print(f"Kolmogorov-Smirnov {calculate_kolmogorov_smirnov(data[:, 2])}")
        print(f"Mardia's Tests {calculate_mardias(data)}")

    main()
