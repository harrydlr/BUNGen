import logging
import os
import matplotlib.pyplot as plt
from netgen import NetworkGenerator
import random
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(filename='bungen_testing_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure logging to capture messages in a list
log_messages = []

def custom_logger(msg):
    log_messages.append(msg)

def generate_list_with_sum(m, n):
    #if m < n:
    #    raise ValueError("The sum (m) must be greater than or equal to the length (n).")

    # Calculate the equal share for each element
    equal_share = m // n
    remainder = m % n
    # Create the list with equal shares
    result = [equal_share] * n
    # Distribute the remainder to the first few elements
    for i in range(remainder):
        result[i] += 1

    return result


# Function to run your script with different parameters
def run_script(rows, columns, block_number, bipartite, P, mu, y_block_nodes_vec, x_block_nodes_vec, fixedConn, link_density, target_error):
    try:
        # Generate network
        M_t, Pij_t, crows_t, ccols_t = NetworkGenerator.generate(rows, columns, block_number, bipartite=bipartite, P=P, mu=mu,
                                                         y_block_nodes_vec= y_block_nodes_vec, x_block_nodes_vec=x_block_nodes_vec,
                                                             fixedConn=fixedConn, link_density=link_density)

        # Save adjacency matrix plot
        title = f'{rows}_{columns}_{block_number}_{bipartite}_{mu}_{fixedConn}_{link_density}'
        plt.imshow(Pij_t, cmap='binary', interpolation='None')
        plt.title(title)
        plot_file_path = f'{title}.png'
        plt.savefig(plot_file_path)
        plt.close()

        logging.info(f'Success: Parameters - {title}')

        # Log success
        custom_logger({'status': 'OK', "target_error":target_error, 'error': None, 'network_plot': plot_file_path, 'parameters': locals()})

    except Exception as e:
        # Log errors
        title = f'{target_error}_{rows}_{columns}_{block_number}_{bipartite}_{mu}_{fixedConn}_{link_density}'
        #logging.info(f'Fail: {title}')
        #error_msg = f'Error: {str(e)}'
        #error_msg = f'Error: Parameters - {title}, Message - {str(e)}'
        #logging.error(error_msg)
        # Log errors
        error_msg = f'Error: {str(e)}'
        custom_logger({'status': 'DOWN', "target_error":target_error, 'error': error_msg, 'network_plot': None, 'parameters': locals()})


def generate_right_random_parameter_set(size):
    # Generate random parameters
    # bipartite variable
    bipartite = random.choice([True, False])
    # rows and cols variable
    if bipartite:
        rows = random.randint(3, size)
        cols = random.randint(3, size)
    else:
        rows = random.randint(3, size)
        cols = rows
    # blocks number variable
    block_number=rows+cols
    while block_number>min(rows, cols):
        block_number = random.randint(1, size)
    # P variable
    if random.choice([True, False]):
        P = random.uniform(0, 1)
    else:
        P = [random.uniform(0, 1) for _ in range(block_number)]
    # mu
    mu = random.uniform(0, 1)
    # y_block_nodes_vec variable
    y_block_nodes_vec = generate_list_with_sum(rows, block_number)
    # x_block_nodes_vec variable
    x_block_nodes_vec = generate_list_with_sum(cols, block_number)
    # fixedConn variable
    fixedConn = False
    # Link density variable
    link_density = random.uniform(0.01, 1)
    #
    target_error = None
    return rows, cols, block_number, bipartite, P, mu, y_block_nodes_vec,  x_block_nodes_vec, fixedConn, link_density, target_error

def generate_wrong_random_parameter_set(size):
    error_list = ["bipartite_var_error","unipartite_error", "size_error", "block_number_error", "P_error", "mu_error", "y_block_nodes_vec",  "x_block_nodes_vec",  "fixedConn_error", "link_density_error"]
    selected_error = np.random.choice(error_list)
    # Generate random parameters
    rows, cols, block_number, bipartite, P, mu, y_block_nodes_vec,  x_block_nodes_vec, fixedConn, link_density, target_error = generate_right_random_parameter_set(size)
    # bipartite variable error
    if selected_error=="bipartite_var_error":
        target_error = "bipartite_var_error"
        bipartite = np.random.choice([np.random.uniform(-1000, 1000), ""])
    # unipartite error
    elif selected_error=="unipartite_error":
        target_error = "unipartite_error"
        bipartite = False
        while rows==cols:
            cols = random.randint(3, size)
    # Network size error
    elif selected_error == "size_error":
        target_error = "size_error"
        aux_size_error= np.random.choice(["col", "row", "colrow"])
        if aux_size_error=="col":
            cols=np.random.choice([np.random.uniform(-2, 2), random.randint(0, 2),[random.randint(1, 10)], str(random.randint(1, 10)), "", True, False])
        elif aux_size_error=="row":
            rows = np.random.choice([np.random.uniform(-2, 2), random.randint(0, 2), [random.randint(1, 10)], str(random.randint(1, 10)), "", True, False])
        else:
            rows = np.random.choice([np.random.uniform(-2, 2), random.randint(0, 2), [random.randint(1, 10)], str(random.randint(1, 10)), "", True, False])
            cols = np.random.choice([np.random.uniform(-1000, -1), random.randint(0, 2), [random.randint(1, 10)], str(random.randint(1, 10)), "", True, False])
    # Blocks number error
    elif selected_error == "block_number_error":
        target_error = "block_number_error"
        block_number = np.random.choice([np.random.uniform(-1000, 0), ["True"], str(random.randint(0, 10)), rows+cols])
    # P variable error
    elif selected_error == "P_error":
        target_error = "P_error"
        if random.choice([True, False]):
            P = np.random.choice([np.random.uniform(-1000, -1), np.random.uniform(2, 1000), ["True"], str(random.randint(0, 1))])
        else:
            aux_block_number = block_number
            while aux_block_number == block_number:
                aux_block_number = random.randint(1, block_number+100)
            P = [random.uniform(0, 1) for _ in range(aux_block_number)]
    # mu variable error
    elif selected_error == "mu_error":
        target_error = "mu_error"
        mu = np.random.choice([random.randint(2, 10), np.random.uniform(-10, -1), "True"])
    # y_block_nodes_vec error
    elif selected_error == "y_block_nodes_vec":
        target_error = "y_block_nodes_vec"
        aux_y_block_nodes_vec = np.random.choice(["row", "block", "rowblock"])
        if aux_y_block_nodes_vec == "row":
            aux_row = rows
            while aux_row <= block_number or aux_row== rows:
                aux_row = random.randint(1, rows+100)
            y_block_nodes_vec = generate_list_with_sum(aux_row, block_number)
        elif aux_y_block_nodes_vec == "block":
            aux_block = block_number
            while aux_block == block_number or aux_block>= rows:
                aux_block = random.randint(1, block_number + 100)
            y_block_nodes_vec = generate_list_with_sum(rows, aux_block)
        else:
            aux_row = rows
            aux_block = block_number
            while aux_row <= block_number or aux_row == rows:
                aux_row = random.randint(1, rows + 100)
            while aux_block == block_number or aux_block >= aux_row:
                aux_block = random.randint(1, block_number + 100)
            y_block_nodes_vec = generate_list_with_sum(aux_row, aux_block)
    # x_block_nodes_vec error
    elif selected_error == "x_block_nodes_vec":
        target_error = "x_block_nodes_vec"
        aux_x_block_nodes_vec = np.random.choice(["col", "block", "colblock"])
        if aux_x_block_nodes_vec == "col":
            aux_col = cols
            while aux_col == cols:
                aux_col = random.randint(1, cols+100)
            x_block_nodes_vec = generate_list_with_sum(aux_col, block_number)
        elif aux_x_block_nodes_vec == "block":
            aux_block = block_number
            while aux_block == block_number:
                aux_block = random.randint(1, block_number + 100)
            x_block_nodes_vec = generate_list_with_sum(cols, aux_block)
        else:
            aux_col = cols
            aux_block = block_number
            while aux_col == cols:
                aux_col = random.randint(1, cols+100)
            while aux_block == block_number:
                aux_block = random.randint(1, block_number + 100)
            x_block_nodes_vec = generate_list_with_sum(aux_col, aux_block)
    # fixedConn error
    elif selected_error == "fixedConn_error":
        target_error = "fixedConn_error"
        fixedConn = np.random.choice([np.random.uniform(-1000, 1000)])
    # Link density error
    elif selected_error == "link_density_error":
        target_error = "link_density_error"
        if fixedConn:
            link_density = np.random.choice([np.random.uniform(-1000, -1), random.uniform(1.1, 10)])
        else:
            link_density = np.random.choice([np.random.uniform(-1000, 0)])
    return rows, cols, block_number, bipartite, P, mu, y_block_nodes_vec,  x_block_nodes_vec, fixedConn, link_density, target_error

# Example usage to generate 100 sets of parameters
n_size = 100  # Adjust as needed
num_sets = 100

# Use this to generate right parameters
#parameter_sets = [generate_right_random_parameter_set(n_size) for _ in range(num_sets)]

# Generate wrong parameters
parameter_sets = [generate_right_random_parameter_set(n_size) for _ in range(num_sets)]

# Run the script for each parameter set
for params in parameter_sets:
    run_script(*params)

# Create a DataFrame from the log messages
df = pd.DataFrame(log_messages)
# Save the DataFrame to an Excel file
excel_file_path = 'wrong_parameter_results.xlsx'
df.to_excel(excel_file_path, index=False)

print(f'Excel file saved at: {os.path.abspath(excel_file_path)}')