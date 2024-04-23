import torch
import matplotlib.pyplot as plt

def load_data(filename):
    """
    Load tensor data from a .pt file.

    Parameters:
    filename (str): The path to the .pt file containing the tensor data.

    Returns:
    torch.Tensor: The loaded tensor.
    """
    return torch.load(filename)

def display_point_cloud(data):
    """
    Display a 2D point cloud using matplotlib.

    Parameters:
    data (torch.Tensor): A tensor containing the point cloud data.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Data must be a 2D tensor with two columns for 2D point visualization.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    plt.title('2D Point Cloud Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def main(filename):
    """
    Main function to load tensor data and display it as a point cloud.

    Parameters:
    filename (str): The path to the .pt file to be loaded and displayed.
    """
    data = load_data(filename)
    display_point_cloud(data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python display_2d.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    main(filename)
