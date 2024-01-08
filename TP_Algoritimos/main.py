import random
import bfs
import a_star

ROBOT_SYMBOL = 'R'
OBJECT_SYMBOL = 'P'
# RECEPTION_SYMBOL = 'G'

def get_matrix_size(dimension):
    return int(input(f"Enter the {dimension} of the matrix: "))

def create_matrix():
    matrix_width = get_matrix_size("width")
    matrix_height = get_matrix_size("height")
    obstacle_density = float(input("Enter obstacle density (0.0 to 1.0): "))

    matrix = [[1 if random.random() < obstacle_density else 0 for _ in range(matrix_width)] for _ in range(matrix_height)]

    matrix[0][8] = ROBOT_SYMBOL
    # matrix[0][0] = RECEPTION_SYMBOL

    object_x, object_y = random.randint(0, matrix_height - 1), random.randint(0, matrix_width - 1)
    while matrix[object_x][object_y] == 1:
        object_x, object_y = random.randint(0, matrix_height - 1), random.randint(0, matrix_width - 1)

    matrix[object_x][object_y] = OBJECT_SYMBOL

    filename = input("Enter the name of the file to save the matrix (.txt): ")
    save_matrix_to_file(matrix, filename)
    return matrix

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

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

def save_matrix_to_file(matrix, filename):
    with open(filename, "w") as f:
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def use_saved_matrix():
    filename = input("Enter the name of the file (.txt): ")
    matrix = read_matrix_from_file(filename)
    if matrix is not None:
        print_matrix(matrix)
    return matrix

def main():
    while True:
        print("\n\t\tRobot Pathfinding Menu\t\t\n")
        print("1. Create Matrix")
        print("2. Use Algorithms")
        print("3. Exit")

        choice = input("Choose a number: ")

        if choice == "1":
            matrix = create_matrix()
            print_matrix(matrix)

        elif choice == "2":
            print("\n\t\tAlgorithm Selection Menu\t\t\n")
            print("1. Breadth-first search")
            print("2. A* search")

            choice2 = input("Choose a number: ")

            if choice2 == "1":
                matrix = use_saved_matrix()
                if matrix is not None:
                    bfs.bfs(matrix)

            elif choice2 == "2":
                matrix = use_saved_matrix()
                if matrix is not None:
                    a_star.a_star(matrix)

            else:
                print("Invalid choice. Please select a valid option.")

        elif choice == "3":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == '__main__':
    main()
