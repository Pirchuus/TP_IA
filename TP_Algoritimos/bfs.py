from collections import deque
import pygame

class Node:
    def __init__(self, x, y, direction=None, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.direction = direction

def is_valid(x, y, matrix):
    return 0 <= x < len(matrix) and 0 <= y < len(matrix) and matrix[x][y] != 1

# def visualize_path(matrix, path, visited):
#     pygame.init()

#     width, height = 800, 800
#     matrix_size = len(matrix)
#     cell_size = width // matrix_size

#     screen = pygame.display.set_mode((width, height))
#     pygame.display.set_caption("Pathfinding Visualization")

#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return

#         screen.fill((255, 255, 255))

#         for i in range(matrix_size):
#             for j in range(matrix_size):
#                 if matrix[i][j] == 1:  # Obstacle
#                     pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
#                 elif matrix[i][j] == 'R':  # Start (Robot)
#                     pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
#                 elif matrix[i][j] == 'P':  # Goal (Object)
#                     pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
#                 elif (i, j) in path:  # Final path (green)
#                     pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
#                 elif (i, j) in visited:  # Explored path (red)
#                     pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
#                 else:
#                     pygame.draw.rect(screen, (255, 255, 255), (j * cell_size, i * cell_size, cell_size, cell_size))

#         pygame.display.flip()

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
    
    # pygame.init()

    # width, height = 800, 800
    # matrix_size = len(matrix)
    # cell_size = width // matrix_size
    
    # screen = pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Pathfinding Visualization")

    queue = deque([start])
    visited = set()
    path_found = False
    total_cost = 0
    last_direction = None

    while queue:
        current = queue.popleft()
        x, y = current.x, current.y
        
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         if not path_found:  # Check if the path has not been found
        #             print("No path found.")
        #         return

        # screen.fill((255, 255, 255))

        # for i in range(matrix_size):
        #     for j in range(matrix_size):
        #         if matrix[i][j] == 1:  # Obstacle
        #             pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif matrix[i][j] == 'R':  # Start (Robot)
        #             pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif matrix[i][j] == 'P':  # Goal (Object)
        #             pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif (i, j) in visited:  # Explored path (red)
        #             pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

        # pygame.display.flip()
        
        if (x, y) == (goal.x, goal.y):
            path = []

            while current:
                path.append((current.x, current.y))

                total_cost += 1  # Count the step

                if current.direction and current.direction != last_direction:
                    total_cost += 1  # Add extra cost for rotation

                last_direction = current.direction
                current = current.parent

            path.reverse()

            if path:
                print(f"Path: {path}")
                print(f"Total Cost: {total_cost}")
                path_found = True
                # visualize_path(matrix, path, visited)

            return path, total_cost

        if (x, y) not in visited:
            visited.add((x, y))

            for dx, dy, direction in [(0, 1, 'right'), (0, -1, 'left'), (1, 0, 'down'), (-1, 0, 'up')]:
                new_x, new_y = x + dx, y + dy
                if is_valid(new_x, new_y, matrix):
                    neighbor = Node(new_x, new_y, direction, parent=current)
                    queue.append(neighbor)

    if not path_found:
        print("No path found.")
        # pygame.quit()
    return None, None