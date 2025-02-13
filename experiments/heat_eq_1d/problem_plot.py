import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['text.usetex'] = True  
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(figsize=(8, 6))

domain_width = 1.0
domain_height = 0.4
insulation_thickness = 0.05

domain_rect = patches.Rectangle(
    (0, 0), domain_width, domain_height,
    edgecolor='black', facecolor='white', zorder=2
)
ax.add_patch(domain_rect)

# Draw top insulation (hatched)
top_insulation = patches.Rectangle(
    (0, domain_height), 
    domain_width, insulation_thickness,
    edgecolor='black', facecolor='white',
    hatch='////', zorder=2
)
ax.add_patch(top_insulation)

# Draw bottom insulation (hatched)
bottom_insulation = patches.Rectangle(
    (0, -insulation_thickness), 
    domain_width, insulation_thickness,
    edgecolor='black', facecolor='white',
    hatch='////', zorder=2
)
ax.add_patch(bottom_insulation)

# Annotate the domain with Q_g
ax.text(0.5, 0.2, r"$Q_g = 10\ \mathrm{W/m^3}$",
        horizontalalignment='center', verticalalignment='center')

# Label top and bottom “Insulated”
ax.text(domain_width/2, domain_height + 1.5*insulation_thickness, "Insulated",
        horizontalalignment='center', verticalalignment='bottom')
ax.text(domain_width/2, -1.5*insulation_thickness, "Insulated",
        horizontalalignment='center', verticalalignment='top')

# Boundary temperatures on left and right
ax.text(-0.05, domain_height/2, r"$T = 0^\circ\mathrm{C}$",
        horizontalalignment='right', verticalalignment='center')
ax.text(domain_width + 0.05, domain_height/2, r"$T = 0^\circ\mathrm{C}$",
        horizontalalignment='left', verticalalignment='center')

# Arrow to show x-direction
ax.annotate("",
            xy=(0.2, 0.2), xytext=(0, 0.2),
            arrowprops=dict(arrowstyle="->", lw=1.0))
ax.text(0.22, 0.19, r"$x$", horizontalalignment='center')

# Arrow with 1 m length
ax.annotate("",
    xy=(0, -0.150), xytext=(1, -0.150),
    arrowprops=dict(arrowstyle="<->", lw=1.0)
)
ax.text(0.5, -0.16, r"1 m", horizontalalignment='center', verticalalignment='top')

# Adjust axes
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.3, 0.5)
ax.set_aspect('equal', 'box')
ax.axis('off')
plt.tight_layout()

plot_path = os.path.join(BASE_DIR, "problem_plot.pdf")
plt.savefig(plot_path, dpi=300)
plt.show()
