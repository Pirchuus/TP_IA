import pygame
from pygame.locals import QUIT
import time

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        
        self.g = 0
        self.h = 0
        self.f = 0
    
    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    dx = abs(node.x - goal.x)
    dy = abs(node.y - goal.y)
    
    return (dx + dy) ** 2

def is_valid(x, y, matrix):
    # Check if the coordinates are within the bounds of the matrix and not an obstacle (1)
    return 0 <= x < len(matrix) and 0 <= y < len(matrix) and matrix[x][y] != 1

def calculate_cell_size(matrix_size, max_window_size):
    max_cell_size = max_window_size // matrix_size
    return min(max_cell_size, 10)

def a_star(matrix):
    start = None
    goal = None
    obstacles = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'R':  # Start (Robot)
                start = Node(i, j)
            elif matrix[i][j] == 'P':  # Goal (Object)
                goal = Node(i, j)
            elif matrix[i][j] == 1:  # Obstacle
                obstacles.append((i, j))

    if start is None or goal is None:
        print("Start or goal not found in the matrix.")
        return

    pygame.init()
    
    cell_size = calculate_cell_size(len(matrix), 700)  # Adjust the cell size as needed
    window_width = len(matrix) * cell_size
    window_height = len(matrix) * cell_size
    matrix_size = len(matrix)
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Pathfinding Visualization")
    
    open_set = [start]
    closed_set = set()
    start_time = time.time()
    path_found = False

    while open_set:
        current = min(open_set, key=lambda node: node.f)
        open_set.remove(current)
        closed_set.add((current.x, current.y))

        x, y = current.x, current.y

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                if not path_found:  # Check if the path has not been found
                    print("No path found.")
                return

        screen.fill((255, 255, 255))

        for i in range(matrix_size):
            for j in range(matrix_size):
                if matrix[i][j] == 1:  # Obstacle
                    pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                elif matrix[i][j] == 'R':  # Start (Robot)
                    pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                elif matrix[i][j] == 'P':  # Goal (Object)
                    pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
                elif (i, j) in closed_set:  # Explored path (red)
                    pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

        pygame.display.flip()
        pygame.time.delay(10)

        if (x, y) == (goal.x, goal.y):
            path = []
            total_cost = 0  # Initialize the total cost variable

            while current:
                path.append((current.x, current.y))
                total_cost += current.g  # Accumulate the cost
                current = current.parent

            path.reverse()

            if path:
                elapsed_time = time.time() - start_time
                print(f"\n\nPath found in {elapsed_time:.6f} seconds.")
                print(f"Total Cost: {total_cost}")  # Print the total cost
                print(f"\nPath: {path}")


                path_found = True

                while True:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            return

                    screen.fill((255, 255, 255))

                    for i in range(matrix_size):
                        for j in range(matrix_size):
                            if matrix[i][j] == 1:  # Obstacle
                                pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                            elif matrix[i][j] == 'R':  # Start (Robot)
                                pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
                            elif matrix[i][j] == 'P':  # Goal (Object)
                                pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
                            elif (i, j) in path:  # Final path (green)
                                pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                            elif (i, j) in closed_set:  # Explored path (red)
                                pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

                    pygame.display.flip()
                    pygame.time.delay(10)

                    # Check if the user closes the window or presses a key to exit the loop
                    for event in pygame.event.get():
                        if event.type == QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            pygame.quit()
                            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y, matrix):
                if (new_x, new_y) not in closed_set:
                    neighbor = Node(new_x, new_y, parent=current)
                    neighbor.g = current.g + 1
                    neighbor.h = heuristic(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h

                    if neighbor not in open_set:
                        open_set.append(neighbor)
                    else:
                        # Update cost if new path is better
                        existing_neighbor = next((n for n in open_set if n == neighbor), None)
                        if existing_neighbor and neighbor.g < existing_neighbor.g:
                            existing_neighbor.g = neighbor.g
                            existing_neighbor.parent = current
                            existing_neighbor.f = existing_neighbor.g + existing_neighbor.h


    if not path_found:
        print("No path found.")
    return None