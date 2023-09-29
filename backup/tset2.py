import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# Initialization
N=300
mean = np.array([0,0])
covariance = np.diag([9,9])
samples = np.random.multivariate_normal(mean, covariance, size=N)

def ground_truth(t):
    slope = 1
    intercept = 0 
    x = 0.5 * t
    y = slope * x + intercept
    return x,y

# Define the increment and the range of x-values
increment = 0.1
x_values = np.arange(0, 5, increment)

# Compute the corresponding y-values for a diagonal line (y = mx + c)
slope = 1.0  # Slope of 1 for a 45-degree diagonal line
intercept = 0  # You can adjust the intercept if needed
y_values = slope * x_values + intercept

# # Create the plot
# plt.plot(x_values, y_values, '-o', color='b')
# # Display the plot
# plt.grid(True)
# plt.show()

sigma=1 # standard deviation
variance = sigma**2
def likelihood(location,m):
    return np.exp(-0.5*(location[0]-m)**2/variance)

m = 0
gt = [0,0]
gt_labels = [gt]
print(samples.shape)
for i in range(15):

    weights = np.apply_along_axis(likelihood, 1, samples, m)
    print("weight",weights.shape)
    # fig = px.scatter(x=samples[:,0], y=samples[:,1], size=weights)
    # fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
    # fig.update_yaxes(range=[-10,30], scaleanchor = "x",scaleratio = 1) # axis equal
    # fig.show()

    # Simulate all samples forward for one second, using 10 Euler steps:
    V=2
    predictions = np.copy(samples)
    
    x = predictions[:,0]
    y = predictions[:,1]
    predictions[:,0] += V * np.cos(np.pi/4)
    predictions[:,1] += V * np.sin(np.pi/4)

    gt[0] = gt[0] + V * np.cos(np.pi/4)
    gt[1] = gt[1] + V * np.sin(np.pi/4)

    gt_labels.append(gt)
    m = gt[0]
    # fig = px.scatter(x=predictions[:,0], y=predictions[:,1], size=weights)
    # fig.update_layout(width = 500, height = 500, title = "Weighted samples from posterior")
    # fig.update_yaxes(range=[-10,30], scaleanchor = "x",scaleratio = 1) # axis equal
    # fig.show()

    # Resample
    sample_indices = np.random.choice(len(samples),p=weights/np.sum(weights),size=N)
    samples = predictions[sample_indices]

    # fig = px.scatter(x=samples[:,0], y=samples[:,1])
    # fig = px.scatter(gt[0],gt[1])
    # fig.update_layout(width = 500, height = 500, title = "Reweighted samples")
    # fig.update_yaxes(range=[-10,30], scaleanchor = "x",scaleratio = 1) # axis equal
    # fig.show()

    plt.plot(samples[:,0], samples[:,1],'*')
    plt.plot(gt[0],gt[1],'o')
    plt.xlabel('x')
    plt.ylabel('sinc(x)')
    plt.title('Plot of the sinc function')
    # Set X and Y axis limits
    plt.xlim(-5, 30)  # Set X-axis limits from 0 to 6
    plt.ylim(-5, 30)  # Set Y-axis limits from 0 to 8
    plt.grid(True)
    plt.show()