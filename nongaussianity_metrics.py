import synthetic_generators
import torch
import scipy

def calculate_moments(points, num_moments=4, eps=1e-8):
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

    num_points = points.shape[0]
    mean = torch.mean(points, dim=0)

    if num_moments == 1:
        return (mean,)

    centered_points = points - mean
    covariance = torch.matmul(centered_points.T, centered_points) / (num_points - 1)

    if num_moments == 2:
        return mean, covariance

    std_dev = torch.std(points, dim=0, unbiased=False)
    skewness = torch.mean((points - mean) ** 3, dim=0) / (std_dev ** 3 + eps)

    if num_moments == 3:
        return mean, covariance, skewness

    kurtosis = torch.mean((points - mean) ** 4, dim=0) / (std_dev ** 4 + eps)

    return mean, covariance, skewness, kurtosis

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
    p_value = 1 - scipy.stats.chi2.cdf(jb_stat, df=2)  # Assuming two degrees of freedom for the test
    
    # For now, returning the JB statistic without a p-value
    return jb_stat, p_value  # Placeholder for p-value



if __name__ == "__main__":

    def main():
        # Example data
        num_points = 1000
        dimensions = 3
        data = synthetic_generators.sample_gaussian(num_points, dimensions)
        
        # Calculate up to the 4th moment
        moments = calculate_moments(data, 4)
        
        print("Mean:\n", moments[0])
        print("Covariance:\n", moments[1])
        print("Skewness:\n", moments[2])
        print("Kurtosis:\n", moments[3])

        print(f"Jarque-Bera {calculate_jarque_bera(num_points, moments[2], moments[3])}")

    main()
