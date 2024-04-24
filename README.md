# Gaussianity Metrics for Point Clouds
This repository provides tools to calculate and visualize metrics that help determine how closely a point cloud resembles a Gaussian distribution.

## Introduction
This repository is dedicated to developing and demonstrating various metrics for determining the non-Gaussianity of point clouds. Many real-world datasets exhibit complex structures that go beyond simple Gaussian distributions. This repository focuses on  techniques for assessing the nongaussianity of point cloud data, providing a toolkit for researchers and developers working in fields like machine learning, computer vision, and data analysis.

## Motivation
Understanding the Gaussianity (or lack thereof) of a point cloud is valuable in several areas of machine learning and data analysis. Some algorithms, such as Independent Component Analysis (ICA), explicitly rely on measures of non-Gaussianity to identify independent components in a dataset, making the assessment of Gaussianity a crucial step in their application. Other algorithms that utilize Gaussianity or non-Gaussianity metrics include:

- **Projection Pursuit**: This technique seeks to find the most "interesting" projections of a high-dimensional dataset, where interestingness is often defined in terms of non-Gaussianity. Metrics like kurtosis or negentropy are used to quantify the non-Gaussianity of the projections.
- **Blind Source Separation (BSS)**: Similar to ICA, various BSS techniques rely on the assumption that the underlying source signals are non-Gaussian. These algorithms seek to recover the original signals from a mixture.
- **Non-Gaussian Component Analysis**: This extension of traditional component analysis methods focuses on finding directions within data that exhibit non-Gaussian characteristics.
- **Anomaly Detection**: Algorithms that detect outliers or anomalies often assume a Gaussian distribution in the dataset. Non-Gaussianity metrics can help identify when such assumptions do not hold, which is crucial for correctly identifying anomalies.
- **Adaptive Filtering**: In signal processing, adaptive filters such as the Least Mean Squares (LMS) algorithm assume Gaussian noise. Non-Gaussian noise characteristics can influence the performance of these algorithms, making Gaussianity metrics important for adjusting filter parameters or choosing alternative filtering strategies.

Moreover, many other algorithms, such as Gaussian Mixture Models, Kalman filters, and some clustering algorithms, assume Gaussian distributions in their underlying models. Assessing the Gaussianity of the data can help validate these assumptions and guide the choice of appropriate algorithms.

Understanding the Gaussianity (or lack thereof) of a point cloud is valuable in several areas of machine learning and data analysis:

- **Algorithm Selection**: Many classic machine learning algorithms (such as PCA, LDA, and Gaussian Mixture Models) have assumptions of Gaussianity in their design. Assessing Gaussianity can guide the choice of appropriate algorithms for your dataset.
- **Data Preprocessing**: If significant deviations from Gaussianity are detected, it might signal the need for transformations (e.g., log-transformation, standardization) to improve the performance of certain algorithms.
- **Dimensionality Reduction and Visualization**: Deviations from Gaussianity can impact the effectiveness of techniques like PCA and t-SNE. These metrics can aid in interpreting results and potentially guide alternative approaches.
- **Outlier Detection**: In some cases, significant departures from Gaussianity within a dataset may highlight outliers or anomalies that warrant further investigation.
- **Manifold Learning**: While many manifold learning algorithms don't explicitly require Gaussianity measures, deviations from Gaussian distributions in local regions of the manifold can impact their performance.
- **Robust Statistics**: Methods designed to be less sensitive to outliers often implicitly consider the possibility of non-Gaussian distributions within the data.

Researchers are actively exploring ways to incorporate Gaussianity assessments into other areas, including feature selection, generative modeling, and even the design of neural network architectures. This project aims to bridge this gap by providing a suite of metrics that more accurately reflect the characteristics of non-Gaussian point clouds. These metrics are designed to enhance the flexibility and accuracy of data analysis pipelines, enabling better decision-making and more nuanced data interpretation. Through this work, we seek to empower researchers and developers with tools to handle and analyze non-Gaussian data more effectively, paving the way for innovative applications across various scientific and engineering disciplines.

## Previous Work
This repository includes the following techniques for assessing Gaussianity in point clouds:

1. **Statistical Moment Analysis**: 
   - **Skewness and Kurtosis**: Measure the asymmetry and tail heaviness of the distribution, respectively. Kurtosis is particularly important for estimating non-Gaussianity as it indicates the presence of outliers or extreme values.
   - **Jarque-Bera Test**: Combines skewness and kurtosis to perform a goodness-of-fit test for Gaussian distributions, assessing whether sample data have the skewness and kurtosis matching a normal distribution.

2. **Normality Tests**: 
Implements a suite of tests to evaluate the conformity of the data distribution to normality:
   - **Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov**: Standard tests that assess the hypothesis that a sample comes from a normally distributed population.
   - **Mardia's, Royston's, Henze-Zirkler**: Advanced tests that provide more robust assessments under various conditions and sample sizes.

3. **Entropy Analysis**: 
   - **Energy Test for Normality and Negentropy**: Evaluate the departure from normality by measuring the statistical independence and randomness within the dataset.
   - **Mutual Information (equivalent to Kullback-Leibler Divergence)**: Quantifies the amount of information lost when approximating the true distribution with a Gaussian model.

## Proposed Techniques
In addition to established methods, we propose two novel techniques to further our understanding of Gaussianity in point clouds:

1. **Mahalanobis Distance Histogram**: Constructs a histogram of the Mahalanobis distances between the points in the dataset and the mean of the fitted Gaussian distribution. This heuristic offers a visual and quantitative metric of how well the data conforms to a Gaussian model.
2. **KL Divergence**: Provides a method to estimate the Kullback-Leibler divergence between the empirical distribution of the point cloud and a theoretical Gaussian distribution. This technique helps in quantifying the deviation from Gaussianity.

## Testing Strategy
To rigorously evaluate the effectiveness of our Gaussianity and non-Gaussianity metrics, we employ a diverse set of synthetic point cloud distributions. These distributions are specifically chosen to challenge the assumptions of Gaussianity and test the robustness of our metrics under various realistic scenarios. Below is a description of the different types of distributions we generate:

### Gaussian Distributions
- **Standard Gaussian**: Single-mode distributions where points are symmetrically distributed around the mean, serving as a baseline for Gaussian behavior.

### Non-Gaussian Distributions
- **Uniform Distribution**: Points are evenly distributed across a defined range, lacking the central concentration characteristic of Gaussian distributions.
- **Multimodal Distribution**: Includes bimodal and trimodal distributions, where data contains multiple distinct peaks, complicating the Gaussian model fit.
- **Heavy-Tailed Distributions**:
  - Cauchy and Levy Distributions: These distributions are known for their heavy tails and outlier presence, significantly differing from the Gaussian distribution's tail behavior.
- **Skewed Distributions**:
  - **Log-Normal and Exponential Distributions**: These distributions are skewed, with a concentration of data points on one side of the mode, unlike the symmetric Gaussian distribution.
- **Clustered Distributions**:
  - **Mixture of Gaussians**: A combination of several Gaussian distributions with different means and variances to create overlapping clusters.
  - **Clusters of Varying Sizes and Densities**: To mimic real-world data complexities in segmentation and clustering tasks.
- **Structured or Geometric Patterns**:
  - **Spiral and Grid Patterns**: To simulate structured data from various scientific and industrial sources.
  - **Doughnut and Spherical Distributions**: These circular and volumetric distributions challenge the metrics by introducing hollow-centered data.
- **Sinusoidal and Periodic Distributions**: For testing the metrics against predictable, periodic variations commonly seen in time-series data.
- **Noise-Infused Gaussian Distributions**:
  - **Gaussian with Impulsive Noise**: Normal data contaminated with sparse, large amplitude outliers to mimic error or anomaly occurrences.

### Testing Methodology
Our approach involves systematically applying our Gaussianity metrics to each of these distributions and evaluating their sensitivity and specificity in identifying non-Gaussian characteristics. We quantify the performance of each metric through:

- **Visual Inspections**: Using plots and visualizations to qualitatively assess how well the metrics highlight non-Gaussian features.
- **Statistical Analysis**: Calculating performance metrics such as error rates, sensitivity, and specificity across different scenarios.
- **Comparative Analysis**: Comparing the results obtained from our metrics against traditional Gaussianity tests to establish benchmarks and improvements.

This comprehensive testing strategy ensures that our tools are validated across a wide spectrum of conditions, providing users with robust, tested methodologies for assessing Gaussianity in their data.

### Working Notes

#### Kurtosis difference in univariate vs multivariate
We have "standard kurtosis" which for an n-dim Gaussian has an expected kurtosis of n*(n+2).

And we have "excess kurtosis" which is "kurtosis - expected kurtosis" centered around 0.

There is a way to calculate excess kurtosis directly without subtracting the expected kurtosis.

Multivariate calculations sometimes use excess kurtosis by default.