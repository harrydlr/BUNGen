import os
import sys
import warnings
warnings.simplefilter('always', UserWarning)

#sys.path.insert(1, './BUNGen-refactoring/')
from config.config import EMPIRICAL_NET_DIR
from netgen import NetworkGenerator
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

def ballcurve(x: float, xi: float):
    """
    function to generate the curve for the nested structure, given a shape
    parameter xi. If xi= 1 is linear.
    input:
    ----------
    x: 1D array, [0,1]
        initial values to be evaluated on the function
    xi: number, >=1
        shape parameter of how stylised is the curve
    output:
    ----------
    y: 1D array, [0,1]
        evaluated function
    """
    return 1 - mt.pow(1 - mt.pow(x, 1 / xi), xi)

def xiConnRelationship(rw: int, cl: int, xi: float) -> int:
    """
     Thi function calculates the connectance of a matrix (assuming B=1) for a given
     nested profile given by the shape parameter xi

     inputs:
    cl: (int)
         column number;
     M: (int)
         row number;
     xi: (float)
         nested profile;

     OUTPUT
     C: (float)
         matrix connectance;
    """
    E = 0  # edge counter
    for i in range(cl):
        x = i / cl  # tessellate
        y = ballcurve(x, xi)
        for j in range(rw):
            if j / rw >= y:
                E += 1
    return E

def interpolate_colors(num_colors):
    """
    Interpolate colors between the given RGB values.
    
    Parameters:
        rgb_values (list): List of RGB values, each value should be a tuple (R, G, B).
        num_colors (int): Number of colors to generate.
    
    Returns:
        list: List of RGB tuples representing the interpolated colors.
    """

    # Example RGB values
    #rgb_values = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    rgb_values = [(72, 33, 115), (30, 155, 138), (194, 223, 35)]

    # Convert RGB values to numpy array for easier manipulation
    rgb_array = np.array(rgb_values)
    
    # Calculate the number of segments between each pair of colors
    num_segments = len(rgb_values) - 1
    
    # Calculate the number of colors in each segment
    colors_per_segment = num_colors // num_segments
    
    # Initialize list to store interpolated colors
    interpolated_colors = []
    
    # Interpolate between adjacent colors
    for i in range(num_segments):
        # Generate intermediate colors using linear interpolation
        for j in range(1, colors_per_segment + 1):
            weight1 = (colors_per_segment - j + 1) / (colors_per_segment + 1)
            weight2 = (j) / (colors_per_segment + 1)
            interpolated_color = tuple((weight1 * rgb_array[i] + weight2 * rgb_array[i+1]).astype(int))
            interpolated_colors.append(interpolated_color)
    
    interpolated_colors.reverse()
    interpolated_colors.insert(0, (1,1,1))
    interpolated_colors.append((75, 75, 75))   # Add gray color
    custom_cmap = ListedColormap(interpolated_colors)
    return custom_cmap

def create_gray_colormap(N):
    if(N>=2):
        colors = [((i/N), (i/N), (i/N)) for i in range(N)]  # Shades of gray from black to white
        colors.remove(colors[N-1])
        #colors.reverse()
        colors.insert(0, (1,1,1))    
        colors.append((0.64, 0.75, 0.92))  # Add bluebell color
    else:
        colors = [(1,1,1)]

    #print(colors)
    cmap_name = f'custom_gray_{N}'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N)
    return custom_cmap
    
# import random

# def generate_random_vector_sum_to_zero(length, N):
#     if N <= 0:
#         raise ValueError("N must be a positive number")
    
#     # Generate a list of random integers between -N and N (inclusive)
#     vector = [random.randint(-N, N) for _ in range(length)]
    
#     # Adjust the vector to ensure it sums to 0
#     while sum(vector) != 0:
#         # Choose a random index
#         index = random.randint(0, length - 1)
        
#         # Choose a random increment value between -vector[index] and -vector[index] to balance the sum
#         increment = random.randint(-vector[index], -vector[index])
        
#         # Update the vector
#         vector[index] += increment
    
#     return vector

def printMatrix(M,Pij,crows,ccols,figName):

    # Number of colors to generate (i.e. num Blocks)
    numBlocks = (len(np.bincount(crows)))    

    # Generate grayscale colors
    #cmap = create_gray_colormap(numBlocks+2)
    
    # Generate interpolated colors
    cmap = interpolate_colors(numBlocks+2)

    crows = np.asarray(crows) + 1
    count_vals_r = np.bincount(crows)
    cum_count_vals_r = np.cumsum(count_vals_r)

    ccols = np.asarray(ccols) + 1
    count_vals_c = np.bincount(ccols)
    cum_count_vals_c = np.cumsum(count_vals_c)

    M[M>0] = cmap.N
    color = 1
    positive_values_mask = M > 0
    for b in range(1,len(cum_count_vals_r)):
        start_row, start_col = cum_count_vals_r[b-1], cum_count_vals_c[b-1]
        end_row, end_col = cum_count_vals_r[b], cum_count_vals_c[b]
        #print(start_row, end_row, start_col, end_col)
        mask = np.zeros_like(M, dtype=bool)
        mask[start_row:end_row, start_col:end_col] = True
        submatrix_positive_values_mask = positive_values_mask & mask
        M[submatrix_positive_values_mask] = color
        color += 1

    M = np.flip(M, axis=0)
    Pij = np.flip(Pij, axis=0)

    # MATPLOTLIB code
    fig = plt.figure(constrained_layout=False)
    gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    ax1.set_title(r'$M$', fontname='Times New Roman')
    ax1.set_aspect(1.0)
    # Hide X and Y axes label marks
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.yaxis.set_tick_params(labelleft=False)

    # draw matrix M
    if(numBlocks != 1):
        ax1.pcolormesh(M, vmin=0, vmax=cmap.N, cmap=cmap, alpha=0.9)
    else:
        # ad hoc colormap for 1-block matrix
        color = (153, 51, 255)  # RGB code (72, 33, 115)
        cmap = ListedColormap(['white', tuple(x / 255 for x in color)])
        ax1.pcolormesh(M, edgecolors='gray', linewidth=0, cmap=cmap)

    # draw matrix Pij
    ax2.set_title(r'$P_{ij}$', fontname='Times New Roman')
    ax2.set_aspect(1.0)
    # Hide X and Y axes label marks
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax2.xaxis.set_tick_params(labelbottom=False)
    ax2.yaxis.set_tick_params(labelleft=False)
    ax2.pcolormesh(Pij, vmin=0, vmax=1, cmap='Greys')

    # draw vertical and horizontal lines for both M and Pij
    count_vals_r = count_vals_r[::-1]
    cum_count_vals_r = np.cumsum(count_vals_r)
    for b in range(numBlocks):
        ax1.axhline(y=cum_count_vals_r[b], color='gray', linewidth=0.5)
        ax2.axhline(y=cum_count_vals_r[b], color='gray', linewidth=0.5)
    for b in range(1,numBlocks):
        ax1.axvline(x=cum_count_vals_c[b], color='gray', linewidth=0.5)
        ax2.axvline(x=cum_count_vals_c[b], color='gray', linewidth=0.5)


    plt.savefig(figName, dpi=720, bbox_inches='tight')

########################################
################# MAIN #################
########################################

# <codecell>
# =============================================================================
# Figure 1: 
# =============================================================================

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 3, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

gen = NetworkGenerator(
        rows=48, 
        columns=48, 
        block_number=1,
        p=0.0,
        mu=0.0,
        y_block_nodes_vec=[48],
        x_block_nodes_vec=[48], 
        bipartite=True, 
        fixedConn=False, 
        link_density=1.5)

Ma, _, _, _ = gen()


gen = NetworkGenerator(
        rows=48, 
        columns=48, 
        block_number=4,
        p=1.0,
        mu=0.0,
        y_block_nodes_vec=[20,16,8,4],
        x_block_nodes_vec=[16,20,4,8], 
        bipartite=True, 
        fixedConn=False, 
        link_density=1.0)

Mb, _, _, _ = gen()


gen = NetworkGenerator(
        rows=48, 
        columns=48, 
        block_number=4,
        p=0.0,
        mu=0.0,
        y_block_nodes_vec=[20,16,8,4],
        x_block_nodes_vec=[16,20,4,8], 
        bipartite=True, 
        fixedConn=False, 
        link_density=1.0)

Mc, _, crows, ccols = gen()

# MATPLOTLIB code
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# Left panel
ax1.set_title('Nested network', fontname='Times New Roman',fontsize=6)
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
ax1.set_ylabel('Nodes of type A', fontname='Times New Roman',fontsize=4)
ax1.set_aspect(1.0)

# ad hoc colormap for 1-block matrix
color = (153, 51, 255)  # RGB code (72, 33, 115)
cmap = ListedColormap(['white', tuple(x / 255 for x in color)])
Ma = np.flip(Ma, axis=0)
ax1.pcolormesh(Ma, edgecolors='gray', linewidth=0, cmap=cmap)

# Number of colors to generate (i.e. num Blocks)
numBlocks = (len(np.bincount(crows)))  

# Generate interpolated colors
cmap = interpolate_colors(numBlocks+2)

crows = np.asarray(crows) + 1
count_vals_r = np.bincount(crows)
cum_count_vals_r = np.cumsum(count_vals_r)

ccols = np.asarray(ccols) + 1
count_vals_c = np.bincount(ccols)
cum_count_vals_c = np.cumsum(count_vals_c)

# Middle panel
Mb[Mb>0] = cmap.N
color = 1
positive_values_mask = Mb > 0
for b in range(1,len(cum_count_vals_r)):
    start_row, start_col = cum_count_vals_r[b-1], cum_count_vals_c[b-1]
    end_row, end_col = cum_count_vals_r[b], cum_count_vals_c[b]
    mask = np.zeros_like(Mb, dtype=bool)
    mask[start_row:end_row, start_col:end_col] = True
    submatrix_positive_values_mask = positive_values_mask & mask
    Mb[submatrix_positive_values_mask] = color
    color += 1
Mb = np.flip(Mb, axis=0)
ax2.set_title('Modular network', fontname='Times New Roman',fontsize=6)
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
ax2.set_xlabel('Nodes of type B', fontname='Times New Roman',fontsize=4)
ax2.set_aspect(1.0)
ax2.pcolormesh(Mb, vmin=0, vmax=cmap.N, cmap=cmap, alpha=0.9)

# Right panel
Mc[Mc>0] = cmap.N
color = 1
positive_values_mask = Mc > 0
for b in range(1,len(cum_count_vals_r)):
    start_row, start_col = cum_count_vals_r[b-1], cum_count_vals_c[b-1]
    end_row, end_col = cum_count_vals_r[b], cum_count_vals_c[b]
    mask = np.zeros_like(Mc, dtype=bool)
    mask[start_row:end_row, start_col:end_col] = True
    submatrix_positive_values_mask = positive_values_mask & mask
    Mc[submatrix_positive_values_mask] = color
    color += 1
Mc = np.flip(Mc, axis=0)
ax3.set_title('In-block nested network', fontname='Times New Roman',fontsize=6)
ax3.get_xaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
ax3.set_aspect(1.0)
ax3.pcolormesh(Mc, vmin=0, vmax=cmap.N, cmap=cmap, alpha=0.9)


fig.savefig('fig1.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure 2: 
# =============================================================================
fi = 3
co = 4
fig, axs = plt.subplots(fi, co, sharex=True, sharey=True)

r= 60
c= 60
Bs = [1,2,4]
ps = [0.0, 0.1, 0.5, 1.0]
mus = [0.0, 0.1, 0.5, 1.0]

# upper panel (row 0 of the figure)
# ad hoc colormap for 1-block matrix
color = (153, 51, 255)  # RGB code (72, 33, 115)
cmap = ListedColormap(['white', tuple(x / 255 for x in color)])
xis = [0.5, 1, 2.0, 4.0]
for idx, xi in enumerate(xis):
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=1,
            p=ps[idx],
            mu=0.0, 
            y_block_nodes_vec=[r],
            x_block_nodes_vec=[c],
            bipartite=True, 
            fixedConn=False, 
            link_density=xi)
    M, _, _, _ = gen()
    
    M = np.flip(M, axis=0)
    axs[0,idx].set_aspect(1.0)
    axs[0,idx].pcolormesh(M, cmap=cmap)
    st = r'$\xi = $' + str(xi) + r', $p = $' + str(ps[idx])
    axs[0,idx].set_title(st, fontname='Times New Roman', fontsize=8)
    axs[0,idx].get_xaxis().set_ticks([])
    axs[0,idx].get_yaxis().set_ticks([])
    if idx==0: 
        axs[0,idx].set_ylabel('B = 1', fontname='Times New Roman', fontsize=8)

# middle panel (row 1 of the figure)
for idx, mu in enumerate(mus): 
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=2,
            p=1.0,
            mu=mu, 
            y_block_nodes_vec=[30,30],
            x_block_nodes_vec=[30,30],
            bipartite=True, 
            fixedConn=False, 
            link_density=1.0)
        
    M, Pij, crows, ccols = gen()

    # Number of colors to generate (i.e. num Blocks)
    numBlocks = (len(np.bincount(crows)))  

    # Generate interpolated colors
    cmap = interpolate_colors(numBlocks+2)

    crows = np.asarray(crows) + 1
    count_vals_r = np.bincount(crows)
    cum_count_vals_r = np.cumsum(count_vals_r)

    ccols = np.asarray(ccols) + 1
    count_vals_c = np.bincount(ccols)
    cum_count_vals_c = np.cumsum(count_vals_c)
    
    M[M>0] = cmap.N
    color = 1
    positive_values_mask = M > 0
    for b in range(1,len(cum_count_vals_r)):
        start_row, start_col = cum_count_vals_r[b-1], cum_count_vals_c[b-1]
        end_row, end_col = cum_count_vals_r[b], cum_count_vals_c[b]
        mask = np.zeros_like(M, dtype=bool)
        mask[start_row:end_row, start_col:end_col] = True
        submatrix_positive_values_mask = positive_values_mask & mask
        M[submatrix_positive_values_mask] = color
        color += 1
    M = np.flip(M, axis=0)
    axs[1,idx].get_xaxis().set_ticks([])
    axs[1,idx].get_yaxis().set_ticks([])
    axs[1,idx].set_aspect(1.0)
    
    st = r'$p = $' + str(1.0) + r', $\mu = $' + str(mu)
    axs[1,idx].set_title(st, fontsize=8)
    if idx == 0:
        axs[1,idx].set_ylabel('B = 2', fontname='Times New Roman', fontsize=8)
    
    axs[1,idx].pcolormesh(M, vmin=0, vmax=cmap.N, cmap=cmap, alpha=0.9)
    
# lower panel (row 2 of the figure)
for idx, mu in enumerate(mus): 
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=4,
            p=ps[idx],
            mu=mu, 
            y_block_nodes_vec=[15,15,15,15],
            x_block_nodes_vec=[15,15,15,15],
            bipartite=True, 
            fixedConn=False, 
            link_density=1.0)
        
    M, Pij, crows, ccols = gen()

    # Number of colors to generate (i.e. num Blocks)
    numBlocks = (len(np.bincount(crows)))  

    # Generate interpolated colors
    cmap = interpolate_colors(numBlocks+2)

    crows = np.asarray(crows) + 1
    count_vals_r = np.bincount(crows)
    cum_count_vals_r = np.cumsum(count_vals_r)

    ccols = np.asarray(ccols) + 1
    count_vals_c = np.bincount(ccols)
    cum_count_vals_c = np.cumsum(count_vals_c)
    
    M[M>0] = cmap.N
    color = 1
    positive_values_mask = M > 0
    for b in range(1,len(cum_count_vals_r)):
        start_row, start_col = cum_count_vals_r[b-1], cum_count_vals_c[b-1]
        end_row, end_col = cum_count_vals_r[b], cum_count_vals_c[b]
        mask = np.zeros_like(M, dtype=bool)
        mask[start_row:end_row, start_col:end_col] = True
        submatrix_positive_values_mask = positive_values_mask & mask
        M[submatrix_positive_values_mask] = color
        color += 1
    M = np.flip(M, axis=0)
    axs[2,idx].get_xaxis().set_ticks([])
    axs[2,idx].get_yaxis().set_ticks([])
    axs[2,idx].set_aspect(1.0)
    
    st = r'$p = $' + str(ps[idx]) + r', $\mu = $' + str(mu)
    axs[2,idx].set_title(st, fontsize=8)
    if idx == 0:
        axs[2,idx].set_ylabel('B = 4', fontname='Times New Roman', fontsize=8)
    
    axs[2,idx].pcolormesh(M, vmin=0, vmax=cmap.N, cmap=cmap, alpha=0.9)    
        
plt.tight_layout()
plt.savefig('fig2.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Table 2: 
# =============================================================================

# Table 2 (top): Nested network, no noise (p = 0)
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=1,
        p=0.0,
        mu=0.0,
        y_block_nodes_vec=[60],
        x_block_nodes_vec=[60], 
        bipartite=True, 
        fixedConn=True, 
        link_density=0.35)

M, Pij, crows, ccols = gen()

printMatrix(M,Pij,crows,ccols,'FigT2a.png')

# Table 2 (middle): Nested network, some noise (p > 0)
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=1,
        p=0.3,
        mu=0.0,
        y_block_nodes_vec=[60],
        x_block_nodes_vec=[60], 
        bipartite=True, 
        fixedConn=True, 
        link_density=0.35)

M, Pij, crows, ccols = gen()

printMatrix(M,Pij,crows,ccols,'FigT2b.png')

# Table 2 (bottom): completely random network
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=1,
        p=1.0,
        mu=1.0, 
        y_block_nodes_vec=[60],
        x_block_nodes_vec=[60],
        bipartite=True, 
        fixedConn=True, 
        link_density=0.4)

M, Pij, crows, ccols = gen()
printMatrix(M,Pij,crows,ccols,'FigT2c.png')

# Table 3 (top): Modular network
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=5,
        p=1.0,
        mu=0.5, 
        y_block_nodes_vec=[12,12,12,12,12],
        x_block_nodes_vec=[12,12,12,12,12],
        bipartite=True, 
        fixedConn=True, 
        link_density=0.1)

M, Pij, crows, ccols = gen()
printMatrix(M,Pij,crows,ccols,'FigT3a.png')

# Table 3 (second row): in-block nested network
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=4,
        p=[0.05, 0.1, 0.05, 0.1],
        mu=0.15, 
        y_block_nodes_vec=[20,15,15,10],
        x_block_nodes_vec=[25,15,10,10],
        bipartite=True, 
        fixedConn=True, 
        link_density=0.15)

M, Pij, crows, ccols = gen()
printMatrix(M,Pij,crows,ccols,'FigT3b.png')

# Table 3 (third row): modular and in-block nested network
gen = NetworkGenerator(
        rows=60, 
        columns=60, 
        block_number=4,
        p=[0.0, 0.1, 0.5, 1.0],
        mu=0.3, 
        y_block_nodes_vec=[20,15,15,10],
        x_block_nodes_vec=[25,15,10,10],
        bipartite=True, 
        fixedConn=True, 
        link_density=0.15)

M, Pij, crows, ccols = gen()

printMatrix(M,Pij,crows,ccols,'FigT3c.png')


# <codecell>
# =============================================================================
# Figure 3:  
# =============================================================================
fi = 3
co = 5

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(fi, co, wspace=0.25, hspace=0.0)

ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[1,0]))
ax.append(fig.add_subplot(gs[2,0]))
ax.append(fig.add_subplot(gs[:,1:]))


# Left panels (matrices)
r= 8
c= 8
Bs = [1,2,4]
ds = [1.0, 0.5, 0.25]
p = 1.0
mu = 0.0
counter = 0
for x in range(fi): 
    B = Bs[x]
    module_vec = [int(r/B)] * B
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=B,
            p=p,
            mu=mu, 
            y_block_nodes_vec=module_vec, # equally-sized blocks
            x_block_nodes_vec=module_vec, # equally-sized blocks
            bipartite=True, 
            fixedConn=True, 
            link_density=1/B)
    
    M, Pij, _, _ = gen()  
    ax[x].imshow(M, cmap='binary', vmin=0, vmax=1)
    ax[x].set(xticks=[], yticks=[])
    ax[x].text(4, 3.5, 'd = ' + str(ds[x]), color='r', va='center', ha='center')


# right panel
num = 500
xs = np.linspace(1, 20, num=num)
ys = 1/xs

# ax.title.set_text(r'$\gamma$ = 2')
ax[3].set_xlabel(r'$B$')
ax[3].set_ylabel(r'$d_{max}$')

#mainDir = '/home/harry/PycharmProjects/BUNGen/Empirical_Net/'
mainDir = EMPIRICAL_NET_DIR
commFold = os.listdir(mainDir)
# CLASS [Pollinators, etc.] LOOP
for comm in commFold:
    
    if comm == ".DS_Store": continue

    sys.stderr.write('WORKING ON ' + str(comm) + '\n')
    #commDir = mainDir+comm
    commDir = os.path.join(mainDir, comm)
    matList = os.listdir(commDir)
    
    # MATRIX LOOP
    for zzz, matName in enumerate(matList):
        
        # LOAD MATRIX
        fname = commDir+"/"+matName
        try:
            mat = np.loadtxt(fname, delimiter=',',dtype='float')
        except (ValueError,FileNotFoundError):
            continue
        
        r, c = mat.shape    
    
        # make it binary
        mat = (mat>0).astype('int')
  
        l0 = np.sum(mat)
        d0 = l0/(r*c)
        sys.stderr.write('\trows = ' + str(r) + '\t')
        sys.stderr.write('cols = ' + str(c) + '\t')
        sys.stderr.write('d = ' + str(d0) + '\n')
            
        cte = d0*np.ones(num)
        cte2 = d0*np.ones(num)
        cte[cte>ys] = -10
        cte2[cte2<ys] = -10

        ax[3].scatter(xs, cte, color='g', s=1, alpha=0.1)
        ax[3].scatter(xs, cte2, color='r', s=1, alpha=0.1)

ax[3].plot(xs, ys, '-k')
ax[3].set(xticks=list(np.arange(min(xs+1), max(xs)+0.05, 2)), yticks=list(np.arange(0, 1.1, 0.1)))
ax[3].text(6, 0.22, s=r'$1/B$', fontsize=16)
ax[3].set_xlim([1-0.05, 20+0.05])
ax[3].set_ylim([0, 1])

plt.savefig('fig3.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure 4 (left panel):  density vs xi
# =============================================================================

fig, ax = plt.subplots()
rs = [4, 8, 10, 20, 40]
xis = np.linspace(0.1,4,20)
d = np.zeros((len(xis)))
for rw in rs:
    cl = rw
    for idx, x in enumerate(xis):
        d[idx] = xiConnRelationship(rw, cl, x)
        d[idx] /= rw*cl
           
    ax.scatter(xis, d, s=rw, label=r'$N=M=$' + str(rw))
    
ax.legend()        
plt.xlabel(r'$\xi$')
plt.ylabel(r'$d$')

plt.savefig('fig4a.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure 4 (right panel):  eccentricity test
# =============================================================================
rs = np.arange(4, 201, 2)
cs = np.flip(rs)
B = [1, 2]
p = 0.0
mu = 0.0
dens = 0.1
reps = 20
fig, ax = plt.subplots()
minabs = 1
maxabs = -1
ecc = rs/cs
ds = np.zeros(len(ecc))
for idx, x in enumerate(rs): 
    sys.stderr.write('Eccentricity = ' + str(ecc[idx]) + '\n')
    r = int(x)
    c = int(cs[idx])
    sys.stderr.write('\tr = ' + str(r) + '\tc = ' + str(c) + '\n')
    bl = int(np.random.choice(B))
    gen = NetworkGenerator(
        rows=r, 
        columns=c, 
        block_number=bl,
        p= np.random.uniform(),
        mu= np.random.uniform(), 
        y_block_nodes_vec= [int(r/bl)] * bl, # equally-sized blocks
        x_block_nodes_vec= [int(c/bl)] * bl, # equally-sized blocks
        bipartite=True, 
        fixedConn=True, 
        link_density=dens)

    _, Pij, _, _ = gen()
    dp = np.zeros(reps)
    eccp = ecc[idx]*np.ones(reps)
    for i in range(reps):
        Mrand = np.array(np.random.uniform(0, 1, size=(r, c)))
        M = (Pij > Mrand).astype(int)
        l = np.sum(M)
        d = l / (r*c)
        if minabs > d: minabs = d
        if maxabs < d: maxabs = d
        ds[idx] += d
        dp[i] = d
        sys.stderr.write('\t\td = ' + str(d) + '\n')
    
        
    ax.scatter(eccp, dp, s=2.5, c='lightgrey')
    ds[idx] /= reps
        
plt.xlabel(r'$N/M$')
plt.ylabel(r'$d$')
plt.plot(ecc, ds, 'og', ms=5)

xs = np.linspace(0.1,10, 100)
ys = dens*np.ones(100)
plt.plot(xs, ys, '-r')
plt.axvline(x=1, color='b', ls='--', lw=1)
plt.axvline(x=0.02, color='r', ls='--', lw=1)
plt.axvline(x=50, color='r', ls='--', lw=1)
plt.xlim([ecc[0]-.01, 100])
#plt.xlim([ecc[0]-.01, ecc[-1]+5])
# plt.ylim([minabs-0.001, maxabs+10])
ax.set_xscale('log')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rect_ax = inset_axes(ax, width=2, height=2, bbox_to_anchor=(0.475, .05, .6, .15),
                   bbox_transform=ax.transAxes, loc="lower left", borderpad=5)
rect_ax.axis("off")
rect=plt.Rectangle((0.1,0.05), 0.2, 0.8, transform=rect_ax.transAxes, fill=False, ec="k")
rect_ax.add_patch(rect)
rect_ax.text(0.0, 0.4, s=r'$N$', fontsize=12)
rect_ax.text(0.155, 0.875, s=r'$M$', fontsize=12)

rect_ax2 = inset_axes(ax, width=2, height=2, bbox_to_anchor=(-0.065, .25, .1, .35),
                   bbox_transform=ax.transAxes, loc="lower left", borderpad=5)
rect_ax2.axis("off")
rect=plt.Rectangle((0.1,0.05), 0.8, 0.2, transform=rect_ax2.transAxes, fill=False, ec="k")
rect_ax2.add_patch(rect)
rect_ax2.text(0.0, 0.125, s=r'$N$', fontsize=12)
rect_ax2.text(0.45, 0.265, s=r'$M$', fontsize=12)

plt.savefig('fig4b.jpg', dpi=720, bbox_inches='tight')
