import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

# Set up the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('Animated Cross-Correlation: Window Function vs PDF', fontsize=14)

# Define x-axis
x = np.linspace(-10, 10, 1000)
dt = x[1] - x[0]

# Define the rectangular function (fixed)
# window_width = 2.0
# window = np.zeros_like(x)
# window[np.abs(x) < window_width / 2] = 1.0

# define the triangle function
window_center = 0
window_width = 4.0
window = np.maximum(0, 1 - np.abs(x - window_center) / (window_width/2))
window = np.where(np.abs(x - window_center) <= window_width/2, window, 0.0)

# Define parameters for moving pdf
sigma = 1.0
positions = np.linspace(-10, 10, 200)  # Positions where pdf will be centered

# Initialize plots
line_window, = ax1.plot(x, window, 'b-', linewidth=2, label='Window Function')
line_pdf, = ax1.plot(x, np.zeros_like(x), 'r-', linewidth=2, label='PDF')
ax1.set_xlim(-10, 10)
ax1.set_ylim(-0.1, 1.1)
ax1.set_ylabel('Amplitude')
ax1.set_title('Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Product plot
line_product, = ax2.plot(x, np.zeros_like(x), 'g-', linewidth=2, label='Product')
ax2.set_xlim(-10, 10)
ax2.set_ylim(-0.1, 0.5)
ax2.set_ylabel('Product')
ax2.set_title('Point-wise Product (Integrand)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Correlation plot
correlation_x = []
correlation_y = []
line_corr, = ax3.plot([], [], 'purple', linewidth=2, label='Cross-correlation')
ax3.set_xlim(-8, 8)
ax3.set_ylim(-0.5, 2.5)
ax3.set_xlabel('Position of PDF')
ax3.set_ylabel('Correlation Value')
ax3.set_title('Cross-correlation vs Position')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Animation function
def animate(frame):
    # Current position of pdf
    pos = positions[frame]
    
    # Create pdf centered at current position

    # normal dist
    #pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - pos) / sigma)**2)
    
    # rayleight dist
    pdf = ((x-pos) / (sigma ** 2)) * np.exp(-(x-pos)**2 / (2 * sigma**2))

    # Update pdf plot
    line_pdf.set_ydata(pdf)
    
    # Calculate point-wise product
    product = window * pdf
    line_product.set_ydata(product)
    
    # Calculate correlation (integral of product)
    correlation_value = np.trapz(product, x)
    
    # Update correlation plot
    correlation_x.append(pos)
    correlation_y.append(correlation_value)
    line_corr.set_data(correlation_x, correlation_y)
    
    # Add text annotation for current correlation value
    ax3.clear()
    ax3.plot(correlation_x, correlation_y, 'purple', linewidth=2, label='Cross-correlation')
    ax3.set_xlim(-8, 8)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_xlabel('Position of PDF')
    ax3.set_ylabel('Correlation Value')
    ax3.set_title('Cross-correlation vs Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95, f'Current correlation: {correlation_value:.3f}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return line_window, line_pdf, line_product, line_corr

# Create animation
anim = FuncAnimation(fig, animate, frames=len(positions), interval=50, blit=False, repeat=True)

# Adjust layout to prevent overlap
plt.tight_layout()

# To save as gif (optional)
anim.save('correlation_animation.gif', writer='pillow', fps=20)

# Show the animation
plt.show()

# Alternative: Using scipy's correlate function for comparison
print("\nUsing scipy.signal.correlate for verification:")
pdf_test = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 0) / sigma)**2)
correlation_scipy = signal.correlate(window, pdf_test, mode='same')
print(f"Scipy correlation at center: {correlation_scipy[len(correlation_scipy)//2]:.3f}")