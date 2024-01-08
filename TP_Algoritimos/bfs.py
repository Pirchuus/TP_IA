from collections import deque
import pygame
from pygame.locals import QUIT
import time

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

def is_valid(x, y, matrix):
    return 0 <= x < len(matrix) and 0 <= y < len(matrix) and matrix[x][y] != 1

def calculate_cell_size(matrix_size, max_window_size):
    max_cell_size = max_window_size // matrix_size
    return min(max_cell_size, 10)

def bfs(matrix):
    start = None
    goal = None
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'R':  # Start (Robot)
                start = Node(i, j)
            elif matrix[i][j] == 'P':  # Goal (Object)
                goal = Node(i, j)

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
    
    queue = deque([start])
    visited = set()
    start_time = time.time()  # Record the start time
    path_found = False

    while queue:
        current = queue.popleft()
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
                elif (i, j) in visited:  # Explored path (red)
                    pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

        pygame.display.flip()

        if (x, y) == (goal.x, goal.y):
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            path.reverse()

            if path:
                for (i, j) in path:
                    visited.add((i, j))

                elapsed_time = time.time() - start_time
                print(f"Path found in {elapsed_time:.6f} seconds.")
                print(f"Path: {path}")

                path_found = True  # Set the flag to indicate path found

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
                            elif (i, j) in visited:  # Explored path (red)
                                pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

                    pygame.display.flip()

            pygame.quit()
            return path

        if (x, y) not in visited:
            visited.add((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if is_valid(new_x, new_y, matrix):
                    neighbor = Node(new_x, new_y, parent=current)
                    queue.append(neighbor)

    print("No path found.")
    pygame.quit()
    return None
