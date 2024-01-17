import random
import bfs
import a_star
import greedy
import pandas as pd
import time
import os


ROBOT_SYMBOL = 'R'
OBJECT_SYMBOL = 'P'
#RECEPTION_SYMBOL = 'G'

# Função para obter o tamanho da matriz
def get_matrix_size(dimension):
    return int(input(f"Enter the {dimension} of the matrix: "))

# Função para criar a matriz
def create_matrix():
    matrix_width = get_matrix_size("width")
    matrix_height = get_matrix_size("height")
    obstacle_density = float(input("Enter obstacle density (0.0 to 1.0): "))

    matrix = [[1 if random.random() < obstacle_density else 0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    matrix[0][8] = ROBOT_SYMBOL

    object_x, object_y = random.randint(0, matrix_height - 1), random.randint(0, matrix_width - 1)
    while matrix[object_x][object_y] == 1:
        object_x, object_y = random.randint(0, matrix_height - 1), random.randint(0, matrix_width - 1)

    matrix[object_x][object_y] = OBJECT_SYMBOL

    filename = input("Enter the name of the file to save the matrix (.txt): ")
    save_matrix_to_file(matrix, filename)
    return matrix

# Função para ler a matriz de um arquivo
def read_matrix_from_file(filename):
    try:
        with open(filename) as file:
            matrix = [[int(num) if num.isdigit() else num for num in line.split()] for line in file]
        return matrix
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Função para imprimir a matriz
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

# Função para guardar a matriz em um arquivo
def save_matrix_to_file(matrix, filename):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

# Função para ler a matriz de um arquivo e imprimir
def use_saved_matrix():
    filename = input("Enter the name of the file (.txt): ")
    matrix = read_matrix_from_file(filename)
    if matrix is not None:
        print_matrix(matrix)
    return matrix

def run_algorithm_and_measure(matrix, algorithm_func, matrix_name="Matrix"):
    start_time = time.time()
    path, total_cost = algorithm_func(matrix)
    elapsed_time = time.time() - start_time

    obstacle_density = round((sum(row.count(1) for row in matrix) / (len(matrix) * len(matrix[0]))) * 100)

    result = {
        'Algorithm': algorithm_func.__name__,
        'Matrix_Name': matrix_name,
        'Obstacle_Density': obstacle_density,
        'Total_Cost': total_cost,
        'Execution_Time': elapsed_time,
        'Path': path
    }

    return result

# Função principal do programa
def main():
    all_results = []

    while True:
        print("\n\t\tRobot Pathfinding Menu\t\t\n")
        print("1. Create Matrix")
        print("2. Use Algorithms")
        print("3. Run All Algorithms on Matrices in 'armazens' Folder")
        print("0. Exit")

        choice = input("Choose a number: ")

        if choice == "1":
            matrix = create_matrix()
            print_matrix(matrix)

        elif choice == "2":
            print("\n\t\tAlgorithm Selection Menu\t\t\n")
            print("1. Breadth-first search")
            print("2. A* search")
            print("3. Greedy search")
            print("0. Back")

            choice2 = input("Choose a number: ")

            if choice2 == "1":
                matrix = use_saved_matrix()
                if matrix is not None:
                    bfs.bfs(matrix)

            elif choice2 == "2":
                matrix = use_saved_matrix()
                if matrix is not None:
                    a_star.a_star(matrix)

            elif choice2 == "3":
                matrix = use_saved_matrix()
                if matrix is not None:
                    greedy.greedy(matrix)

            elif choice2 == "0":
                continue

            else:
                print("Invalid choice. Please select a valid option.")

        elif choice == "3":
            matrices_folder = "Armazens"
            matrix_files = [f for f in os.listdir(matrices_folder) if os.path.isfile(os.path.join(matrices_folder, f))]

            for matrix_file in matrix_files:
                matrix_path = os.path.join(matrices_folder, matrix_file)
                matrix = read_matrix_from_file(matrix_path)
                if matrix is not None:
                    all_results.append(run_algorithm_and_measure(matrix, bfs.bfs, matrix_file))
                    all_results.append(run_algorithm_and_measure(matrix, a_star.a_star, matrix_file))
                    all_results.append(run_algorithm_and_measure(matrix, greedy.greedy, matrix_file))
                    
            export_results_to_excel(all_results)

        elif choice == "0":
            all_results = []
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

            
def export_results_to_excel(results):
    df = pd.DataFrame(results)
    print(df)
    
    # Format the 'Execution_Time' column as seconds and milliseconds
    df['Execution_Time'] = pd.to_datetime(df['Execution_Time'], unit='s').dt.strftime('%S,%f')[:-3]
    
    # Export the DataFrame to Excel
    df.to_excel('pathfinding_results.xlsx', sheet_name='Results', index=False)
    print("Results exported to 'pathfinding_results.xlsx'.")


if __name__ == '__main__':
    main()