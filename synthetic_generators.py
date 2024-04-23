import os
import torch


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
    if scale is not None:
        result.mul_(scale)
    if covariance is not None:
        L = torch.linalg.cholesky(covariance)
        result.matmul_(L)
    if mean is not None:
        result.add_(mean)
    return result


def sample_skew(num_points, dimension, mean=None, covariance=None, skew=None):
    """
    Generate points from a multivariate skew-normal distribution.

    Parameters:
    num_points (int): Number of points to generate.
    dimension (int): Dimensionality of each data point.
    mean (torch.Tensor | None): Mean of the distribution. Defaults to a vector of zeros.
    covariance (torch.Tensor | None): Covariance matrix of the distribution. Defaults to the identity matrix.
    skew (torch.Tensor | None): Skewness vector of the distribution. Defaults to a vector of zeros (normal distribution).

    Returns:
    torch.Tensor: Tensor of sampled points.
    """

    # Generate Gaussian samples
    result = torch.randn(num_points, dimension)

    if covariance is not None:
        L = torch.linalg.cholesky(covariance)
        result.matmul_(L)

    # Apply skew
    # Omega is the cumulative distribution function of the normal distribution evaluated at each point
    if skew is not None:
        omega = torch.distributions.Normal(0, 1).cdf(skew * result)
        result.add_(skew * omega)

    if mean is not None:
        result.add_(mean)

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
    cauchy_dist = torch.distributions.Cauchy(
        loc=0, scale=scale
    )  # loc is always zero for standard Cauchy
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
    v = torch.distributions.Exponential(1).sample(
        (num_points, dimensions)
    )  # Exponential distribution
    levy_samples = torch.sqrt(scale / v) * torch.cos(2 * torch.pi * u)
    return levy_samples


def sample_exponential(num_points, dimensions, scale=1.0):
    """
    Generate points from an exponential distribution for each dimension independently.

    Parameters:
    num_points (int): The number of samples to generate.
    dimensions (int): The number of dimensions for each sample.
    scale (float): The scale parameter for the exponential distribution, defaults to 1.0.
                    This is the inverse of the rate parameter λ (lambda).

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimensions) containing the sampled points.
    """
    # Create an exponential distribution with the given scale
    exp_dist = torch.distributions.Exponential(rate=1.0 / scale)

    # Sample from this distribution
    samples = exp_dist.sample((num_points, dimensions))

    return samples


def sample_sphere(num_points, dimensions, origin=None, radius=1.0):
    """
    Generate points uniformly distributed on the surface of a sphere with specified origin and radius.

    Parameters:
    num_points (int): The number of points to generate.
    dimensions (int): The dimensionality of the space in which the sphere exists.
    origin (torch.Tensor | None): The center of the sphere in the given space. Defaults to the zero vector.
    radius (float): The radius of the sphere. Defaults to 1.0.

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimensions) containing the points on the sphere surface.
    """
    # Generate random Gaussian points
    points = torch.randn(num_points, dimensions)

    # Normalize each point to be on the surface of the unit sphere
    norms = torch.norm(points, p=2, dim=1, keepdim=True)
    results = points / norms

    # Apply radius and origin
    results.mul_(radius)
    if origin is not None:
        results.add_(origin)
    
    return results

def sample_sinusoidal(num_points, dimensions, frequencies=None, phases=None, amplitudes=None, bounds=None):
    """
    Generate points following a sinusoidal pattern in each dimension, with specified amplitudes, frequencies, phases, and bounds.

    Parameters:
    num_points (int): The number of points to generate.
    dimensions (int): The number of dimensions, each will follow a sinusoid.
    frequencies (list | None): Optional list of frequencies for each dimension. Defaults to 1 for all dimensions.
    phases (list | None): Optional list of phase shifts for each dimension. Defaults to 0 for all dimensions.
    amplitudes (list | None): Optional list of amplitudes for each dimension. Defaults to 1 for all dimensions.
    bounds (list of tuples | None): Optional list of tuples, where each tuple contains the start and end of the range
                                    for each dimension. Defaults to (0, 2π) for all dimensions.

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimensions) containing the sinusoidal points.
    """
    if frequencies is None:
        frequencies = [1.0] * dimensions
    if phases is None:
        phases = [0.0] * dimensions
    if amplitudes is None:
        amplitudes = [1.0] * dimensions
    if bounds is None:
        bounds = [(0, 2 * torch.pi)] * dimensions

    # Initialize the output tensor
    samples = torch.zeros(num_points, dimensions)

    # Create time steps based on bounds
    for i in range(dimensions):
        start, end = bounds[i]
        t = torch.linspace(start, end, num_points)

        # Generate the sinusoidal pattern for the dimension
        samples[:, i] = amplitudes[i] * torch.sin(frequencies[i] * t + phases[i])

    return samples

def sample_impulse_noise(num_points, dimensions, noise_prob, noise_level=1.0):
    """
    Generate impulse noise to be added to any distribution. The noise consists of sparse,
    large amplitude values occurring with a given probability.

    Parameters:
    num_points (int): Number of samples to generate.
    dimensions (int): Dimensionality of each sample.
    noise_prob (float): Probability of noise affecting a given point (0 <= noise_prob <= 1).
    noise_level (float): The amplitude of the noise (default is 1.0, can be adjusted).

    Returns:
    torch.Tensor: A tensor of shape (num_points, dimensions) containing the impulse noise.
    """
    # Initialize the noise tensor
    result = torch.zeros(num_points, dimensions)

    # Determine which elements will receive the impulse noise
    noise_mask = torch.rand(num_points, dimensions) < noise_prob

    # Apply noise where the mask is True
    # Noise can be positive or negative; here, we assume it's symmetric around zero
    result[noise_mask] = noise_level * torch.sign(torch.rand_like(result[noise_mask]) - 0.5)

    return result

if __name__ == "__main__":

    def main():
        os.makedirs("synthetic_data", exist_ok=True)

        # Specify the sample size
        num_points = 1000
        dimensions = 2

        samples = sample_gaussian(num_points, dimensions)
        torch.save(samples, "synthetic_data/sample_gaussian.pt")

        samples = sample_skew(num_points, dimensions, skew=torch.tensor([1, -1]))
        torch.save(samples, "synthetic_data/sample_skew.pt")

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
            sample_uniform(
                num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])
            ),
            sample_gaussian(num_points, dimensions=1),
        )
        torch.save(samples, "synthetic_data/sample_mixed_dim.pt")

        samples = merge_sets(
            sample_uniform(
                num_points, dimensions=2, bounds=torch.tensor([[-1, -10], [1, 10]])
            ),
            sample_uniform(
                num_points, dimensions=2, bounds=torch.tensor([[-10, -1], [10, 1]])
            ),
        )
        torch.save(samples, "synthetic_data/sample_cross.pt")

        samples = merge_dimensions(
            sample_uniform(
                num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])
            ),
            sample_cauchy(num_points, dimensions=1),
        )
        torch.save(samples, "synthetic_data/sample_cauchy.pt")

        samples = merge_dimensions(
            sample_uniform(
                num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])
            ),
            sample_levy(num_points, dimensions=1),
        )
        torch.save(samples, "synthetic_data/sample_levy.pt")

        samples = merge_dimensions(
            sample_uniform(
                num_points, dimensions=1, bounds=torch.tensor([[-10], [10]])
            ),
            sample_exponential(num_points, dimensions=1),
        )
        torch.save(samples, "synthetic_data/sample_exponential.pt")

        samples = sample_sphere(num_points, dimensions)
        torch.save(samples, "synthetic_data/sample_sphere.pt")

        samples = sample_sinusoidal(num_points, dimensions, [1, 2], [0, torch.pi/2], [1, 0.5], [(0, torch.pi), (0, 4*torch.pi)])
        torch.save(samples, "synthetic_data/sample_sinusoidal.pt")

        samples = sample_sphere(num_points, dimensions) + sample_impulse_noise(num_points, dimensions, noise_prob= 0.1, noise_level= 0.1)
        torch.save(samples, "synthetic_data/sample_impulse.pt")


    main()
