import numpy as np
import plotly.express as px

# Initialization
N=300
mean = np.array([0,0])
covariance = np.diag([9,9])
samples = np.random.multivariate_normal(mean, covariance, size=N)
print("samples",samples[0])
fig = px.scatter(x=samples[:,0], y=samples[:,1])
fig.update_layout(width = 500, height = 500, title = "Samples from prior")
fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
fig.show()

sigma=1 # standard deviation
variance = sigma**2
def likelihood(location): 
    return np.exp(-0.5*(location[0]-m)**2/variance)

gt = np.array([1,0])
gt_tracked = [gt]
m = 3
for i in range(3):

    # m = m + 2 # measurement

    weights = np.apply_along_axis(likelihood, 1, samples)

    # fig = px.scatter(x=samples[:,0], y=samples[:,1], size=weights)
    # fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
    # fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
    # fig.show()

    # Simulate all samples forward for one second, using 10 Euler steps:
    V=2
    predictions = np.copy(samples)
    for i in range(10):
        x = predictions[:,0]
        y = predictions[:,1]
        norm = np.sqrt(x**2 + y**2)
        predictions[:,0] -= 0.1*y*V/norm
        predictions[:,1] += 0.1*x*V/norm

    for i in range(10):
        gtx = gt[0]
        gty = gt[1]
        norm = np.sqrt(gtx**2 + gty**2)
        gtx -= 0.1*gty*V/norm
        gty += 0.1*gtx*V/norm

    gt[0] = gtx
    gt[1] = gty
    
    # fig = px.scatter(x=predictions[:,0], y=predictions[:,1], size=weights)
    # fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
    # fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
    # fig.show()

    # Resample
    sample_indices = np.random.choice(len(samples),p=weights/np.sum(weights),size=N)
    samples = predictions[sample_indices]

    fig = px.scatter(x=samples[:,0], y=samples[:,1])
    fig.update_layout(width = 500, height = 500, title = "Reweighted samples")
    fig.update_yaxes(range=[-10,10], scaleanchor = "x",scaleratio = 1) # axis equal
    fig.show()

    gt_tracked.append(gt)

# c

# import matplotlib.pyplot as plt

# # Create a basic line plot
# plt.plot(gt_tracked[:,0], gt_tracked[:,1])

# # Add labels and a title
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Simple Line Plot')

# # Display the plot
# plt.show()
