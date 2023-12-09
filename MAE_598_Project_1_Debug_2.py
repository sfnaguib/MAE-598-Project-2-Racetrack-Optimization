import torch
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# New parameters for acceleration and braking
acceleration = 11.0  # acceleration in m/s^2
deceleration = -36.0  # deceleration in m/s^2
max_speed = 85.0  # maximum speed in m/s

# Assuming the track is circular with a center at (0, 0) for simplicity
track_center = (0, 0)

# Function to calculate the distance from the car to the center of the track
def distance_to_center(car_position, track_center):
    return np.sqrt((car_position[0] - track_center[0])**2 + (car_position[1] - track_center[1])**2)

# Function to determine if the car should start decelerating
def should_decelerate(car_position, track_center, track_radius, current_speed, deceleration):
    distance_to_boundary = track_radius - distance_to_center(car_position, track_center)
    stopping_distance = (current_speed**2) / (2 * abs(deceleration))
    return stopping_distance >= distance_to_boundary

# Parameters for simulation
wheelbase = 2.5  # wheelbase of the car in meters
dt = 0.1  # time step in seconds
num_steps = 325  # number of steps in the simulation
track_radius = 50  # radius of the track in meters
track_width = 10.0  # width of the track in meters
learning_rate = 0.0008  # learning rate for the optimizer
iterations = 500  # number of optimization iterations
alpha = 15  # weight for the speed factor in the loss function

# Initialize the steering angles as a PyTorch tensor with small random values
steering_angles = torch.randn((num_steps,), device=device) * 0.01
steering_angles.requires_grad = True

# Define the optimizer using the Adam algorithm
optimizer = torch.optim.Adam([steering_angles], lr=learning_rate)

# List to store the loss and path length at each iteration for plotting
loss_history = []
path_length_history = []

# Simulation loop
for i in range(iterations):
    optimizer.zero_grad()

    # Initialize state variables (x, y, theta) and current speed
    x = torch.tensor([2 * track_radius], dtype=torch.float32, device=device)
    y = torch.tensor([0.0], dtype=torch.float32, device=device)
    theta = torch.tensor([np.pi / 2], dtype=torch.float32, device=device)
    current_speed = torch.tensor([0.0], dtype=torch.float32, device=device)  # starting speed as a tensor

    path_length = 0.0  # Total path length is a scalar since it's not a parameter we're optimizing directly
    penalty = 0.0  # Total penalty is a scalar for the same reason
    speeds = []  # List to store the scalar values of speed at each step

    # Simulate the car's trajectory
    for step in range(num_steps):
        car_position = (x.item(), y.item())  # Current car position as a tuple
        if should_decelerate(car_position, track_center, track_radius, current_speed.item(), deceleration):
            current_speed = torch.clamp(current_speed + deceleration * dt, min=0)
        else:
            if current_speed.item() < max_speed:
                current_speed = torch.clamp(current_speed + acceleration * dt, max=max_speed)
        speeds.append(current_speed.item())  # Append the current speed as a scalar value

        # Update car position and orientation
        new_theta = theta + torch.tan(steering_angles[step]) * current_speed / wheelbase * dt
        dx = torch.cos(new_theta) * current_speed * dt
        dy = torch.sin(new_theta) * current_speed * dt
        new_x = x + dx
        new_y = y + dy
        path_length = path_length + torch.sqrt(dx**2 + dy**2)  # Update path length as a scalar

        # Calculate penalty for being outside the track boundaries
        distance_from_center = torch.sqrt((new_x - track_radius)**2 + new_y**2).item()
        if distance_from_center > track_radius + track_width / 2:
            penalty += distance_from_center - torch.relu(track_radius + track_width / 2)
        if distance_from_center < track_radius - track_width / 2:
            penalty += torch.relu(track_radius - track_width / 2) - distance_from_center

        # Update state variables for next step
        x, y, theta = new_x, new_y, new_theta

    # Create a tensor from the list of speeds
    speed_tensor = torch.tensor(speeds, dtype=torch.float32, device=device)

    # Compute the total loss
    speed_factor = torch.mean(speed_tensor) / max_speed
    loss = torch.tensor(path_length, device=device, requires_grad=True) + penalty * 7.5 + alpha * speed_factor
    loss.backward()
    optimizer.step()

    # Print the loss for monitoring
    print(f'Iteration {i+1}, Loss: {loss.item()}, Path Length: {path_length}, Penalty: {penalty}, Speed Factor: {speed_factor.item()}')

    # Store the loss and path length for plotting
    loss_history.append(loss.item())
    path_length_history.append(path_length)

    # Update the steering angles for the next iteration
    steering_angles = steering_angles.detach().clone().requires_grad_(True)

    # Reset the optimizer with the updated steering angles
    optimizer = torch.optim.Adam([steering_angles], lr=learning_rate)


# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Loss over iterations')
plt.xlabel('Iteration number')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration number')
plt.legend()
plt.grid(True)
plt.show()

# Plot the loss history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss over iterations')
plt.xlabel('Iteration number')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration number')
plt.legend()
plt.grid(True)

# Plot the path length history
plt.subplot(1, 2, 2)
plt.plot(path_length_history, label='Path Length over iterations')
plt.xlabel('Iteration number')
plt.ylabel('Path Length')
plt.title('Path Length vs. Iteration number')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Detach the optimized steering angles for plotting
optimized_angles = steering_angles.detach().cpu().numpy()

# Plot the optimized path
x = 2 * track_radius
y = 0.0
theta = np.pi / 2
x_vals, y_vals = [x], [y]

# Start plotting from the starting point
starting_point = (2 * track_radius, 0.0)

for angle in optimized_angles:
    theta += np.tan(angle) * current_speed / wheelbase * dt
    x += np.cos(theta) * current_speed * dt
    y += np.sin(theta) * current_speed * dt
    x_vals.append(x)
    y_vals.append(y)

# Draw the track limits
inner_bound = plt.Circle((track_radius, 0), track_radius - track_width / 2, color='gray', fill=False)
outer_bound = plt.Circle((track_radius, 0), track_radius + track_width / 2, color='gray', fill=False)

# Plotting the figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(inner_bound)
ax.add_artist(outer_bound)
ax.plot(x_vals, y_vals, label='Optimized Path', color='blue')
ax.plot(*starting_point, 'go', label='Starting Point')
ax.set_xlim([0, track_radius * 2])
ax.set_ylim([-track_radius, track_radius])
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Optimized Path and Track for a 90-Degree Turn')
ax.legend()
ax.axis('equal')
plt.show()