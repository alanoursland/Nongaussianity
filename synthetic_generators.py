import os
import torch


def sample_gaussian(num_points, dimensions=1, mean=None, covariance=None, scale=None):
    """
    Generate points from a Gaussian distribution with specified mean and covariance.

    Parameters:
    num_points (int): Number of points to generate.
    dimensions (int): Dimensionality of each data point (default is 1).
    mean (torch.Tensor | None): Mean of the Gaussian distribution. Defaults to a vector of zeros.
    covariance (torch.Tensor | None): Covariance matrix of the Gaussian distribution. Defaults to the identity matrix.
    scale (torch.Tensor | None): Scale tensor applied element-wise to the output. Defaults to a vector of ones.

    Returns:
    torch.Tensor: Tensor of sampled points.
    """
    # Sample from the Gaussian distribution
    result = torch.randn(num_points, dimensions)
    # Apply the mean and covariance
    if covariance is not None:
        L = torch.linalg.cholesky(covariance)
        result.matmul_(L)
    if mean is not None:
        result.add_(mean)
    if scale is not None:
        result.mul_(scale)
    return result


def sample_uniform(num_points, dimensions, bounds=None):
    """
    Generate points from a uniform distribution within specified bounds for each dimension.

    Parameters:
    num_points (int): Number of points to generate.
    dimensions (int): Dimensionality of each data point.
    bounds (torch.Tensor | None): Tensor of shape (2, dimensions) where the first row contains the minimum
                                  values and the second row contains the maximum values for each dimension.
                                  If None, defaults to [0, 1] for all dimensions.

    Returns:
    torch.Tensor: Tensor of sampled points.
    """
    # Generate random samples from 0 to 1, then scale and shift them
    result = torch.rand(num_points, dimensions)
    if bounds is not None:
        range_vals = bounds[1, :] - bounds[0, :]
        min_vals = bounds[0, :]
        result.mul_(range_vals)
        result.add_(min_vals)
    return result


def merge_sets(*samples):
    """
    Add samples from two sets of size n and m, with the same dimeensions, to form a single set of size n+m.

    Parameters:
    *samples: An arbitrary number of torch.Tensor objects to concatenate.

    Returns:
    torch.Tensor: The concatenated set of samples.
    """
    return torch.cat(samples, dim=0)


def merge_dimensions(*samples):
    """
    Add samples from two sets of dimension j and k, with the same number of samples, to form a single set of size n with dimensions j+k.

    Parameters:
    *samples: An arbitrary number of torch.Tensor objects to concatenate.

    Returns:
    torch.Tensor: The concatenated set of samples.
    """
    return torch.cat(samples, dim=1)


def sample_cauchy(num_points, dimensions, scale=1.0):
    """
    Generate samples from a Cauchy distribution for each dimension.

    Parameters:
    num_points (int): The number of samples to generate.
    dimensions (int): The number of dimensions for each sample.
    scale (float): The scale parameter of the Cauchy distribution, defaults to 1.0.

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimension) containing the sampled points.
    """
    cauchy_dist = torch.distributions.Cauchy(loc=0, scale=scale)  # loc is always zero for standard Cauchy
    result = cauchy_dist.sample((num_points, dimensions))
    return result


def sample_levy(num_points, dimensions, scale=1.0):
    """
    Generate samples from a Lévy distribution for each dimension. The Lévy distribution is
    not directly supported in PyTorch, so this function uses a transformation method.

    Parameters:
    num_points (int): The number of samples to generate.
    dimensions (int): The number of dimensions for each sample.
    scale (float): The scale parameter for the Lévy distribution, defaults to 1.0.

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimension) containing the sampled points.
    """
    u = torch.rand(num_points, dimensions)  # Uniform distribution
    v = torch.distributions.Exponential(1).sample((num_points, dimensions))  # Exponential distribution
    levy_samples = torch.sqrt(scale / v) * torch.cos(2 * torch.pi * u)
    return levy_samples


def sample_skewed(num_points, dimension, distribution_type="lognormal", scale=1.0):
    pass


def sample_clustered(
    num_points, dimension, num_clusters, means, covariances, cluster_sizes
):
    pass


def sample_geometric(num_points, dimension, pattern_type="spiral", **pattern_params):
    pass


def sample_trimodal(num_points, dimensions):
    """Generate points from a trimodal distribution."""
    pass


def sample_log_normal(num_points, dimensions):
    """Generate points from a log-normal distribution."""
    pass


def sample_exponential(num_points, dimensions):
    """Generate points from an exponential distribution."""
    pass


def sample_gaussian_mixture(num_points, dimensions, num_components):
    """Generate points from a mixture of Gaussian distributions."""
    pass


def sample_varying_clusters(num_points, dimensions, num_clusters):
    """Generate points from clusters of varying sizes and densities."""
    pass


def sample_spiral(num_points, dimensions):
    """Generate points in a spiral pattern."""
    pass


def sample_grid(num_points, dimensions):
    """Generate points in a grid pattern."""
    pass


def sample_doughnut(num_points, dimensions):
    """Generate points in a doughnut (torus) shape."""
    pass


def sample_sphere(num_points, dimensions):
    """Generate points on the surface of a sphere."""
    pass


def sample_sinusoidal(num_points, dimensions):
    """Generate points from a sinusoidal distribution."""
    pass


def sample_periodic(num_points, dimensions):
    """Generate points from a periodic distribution."""
    pass


def sample_gaussian_with_impulse_noise(num_points, dimensions, noise_prob):
    """Generate points from a Gaussian distribution with impulsive noise."""
    pass


def add_gaussian_noise(data, mean=0.0, std=0.1):
    pass


if __name__ == "__main__":

    def main():
        os.makedirs("synthetic_data", exist_ok=True)

        # Specify the sample size
        num_points = 1000
        dimensions = 2

        samples = sample_gaussian(num_points, dimensions)
        torch.save(samples, "synthetic_data/sample_gaussian.pt")

        samples = sample_uniform(
            num_points, dimensions, bounds=torch.tensor([[0, -1], [1, 2]])
        )
        torch.save(samples, "synthetic_data/sample_uniform.pt")

        samples = merge_sets(
            sample_gaussian(num_points // 2, dimensions, mean=torch.tensor([3, 0])),
            sample_gaussian(num_points // 2, dimensions, mean=torch.tensor([-3, 0])),
        )
        torch.save(samples, "synthetic_data/sample_bimodal.pt")

        samples = merge_dimensions(
            sample_uniform(num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])),
            sample_gaussian(num_points, dimensions=1),
        )
        torch.save(samples, "synthetic_data/sample_mixed_dim.pt")

        samples = merge_sets(
            sample_uniform(num_points, dimensions=2, bounds=torch.tensor([[-1, -10], [1, 10]])),
            sample_uniform(num_points, dimensions=2, bounds=torch.tensor([[-10, -1], [10, 1]]))
        )
        torch.save(samples, "synthetic_data/sample_cross.pt")

        # samples = sample_cauchy(num_points, dimensions)
        samples = merge_dimensions(
            sample_uniform(num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])),
            sample_cauchy(num_points, dimensions=1))
        torch.save(samples, "synthetic_data/sample_cauchy.pt")

        # samples = sample_levy(num_points, dimensions)
        samples = merge_dimensions(
            sample_uniform(num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])),
            sample_levy(num_points, dimensions=1))
        torch.save(samples, "synthetic_data/sample_levy.pt")


    main()
