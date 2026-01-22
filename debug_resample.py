
import numpy as np

def resample_polyline(points, num_points=32):
    """
    Resample a polyline to a fixed number of points using linear interpolation.
    points: (N, 3)
    """
    if len(points) < 2:
        return np.zeros((num_points, 3))

    # Calculate cumulative distance
    dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dists[-1]
    
    # Generate target distances
    target_dists = np.linspace(0, total_len, num_points)
    
    # Interpolate
    new_points = np.zeros((num_points, 3))
    for i in range(3):
        new_points[:, i] = np.interp(target_dists, cum_dists, points[:, i])
        
    return new_points

# Test case from JSON
p = np.array([
    [5.105637884543179, 2.384930013022195, 0.11258601154065022],
    [38.37603971318735, 2.4223379629451127, -0.014657882824962463]
])

out = resample_polyline(p, 32)
print("Input shape:", p.shape)
print("Output shape:", out.shape)
print("Start:", out[0])
print("End:", out[-1])
print("Mid:", out[16])
