import numpy as np

def generate_2d_gaussian(from_xy=-0.5, to_xy=0.5, n_samples=100):
    x = np.linspace(-0.5, 0.5, n_samples)
    y = np.linspace(-0.5, 0.5, n_samples)
    
    targets = np.subtract(np.exp(-(np.add(np.square(x), np.square(y))/10)), 10)
    patterns = np.array([(x[i], y[i]) for i in range(len(x))])

    return patterns, targets
    

def main():
    generate_2d_gaussian()

if __name__ == '__main__':
    main()