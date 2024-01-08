import pygame

# Node class
class Node:
    def __init__(self, position, parent=None, direction=None):
        self.position = position  
        self.parent = parent  
        self.direction = direction
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

# Heuristic function
def heuristic(node, goal):
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

# Function to check if the movement is valid
def is_valid(x, y, matrix):
    return 0 <= x < len(matrix) and 0 <= y < len(matrix) and matrix[x][y] != 1

# Main function of the greedy algorithm
def greedy(matrix):
    start = None
    goal = None
    obstacles = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 'R':  # Start (Robot)
                start = Node((i, j))
            elif matrix[i][j] == 'P':  # Goal (Object)
                goal = Node((i, j))
            elif matrix[i][j] == 1:  # Obstacle
                obstacles.append((i, j))

    if start is None or goal is None:
        print("Start or goal not found in the matrix.")
        return
    
    # pygame.init()
    # # Pygame window settings
    # width, height = 800, 800
    # matrix_size = len(matrix)
    # cell_size = width // matrix_size
    
    # screen = pygame.display.set_mode((width, height))
    # pygame.display.set_caption("Pathfinding Visualization")
    
    open_set = [start]
    closed_set = set()
    path_found = False
    total_cost = 0

    # Main loop of the algorithm
    while open_set:
        current = min(open_set, key=lambda node: node.f)
        open_set.remove(current)
        closed_set.add(current.position)

        x, y = current.position
        

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         if not path_found:
        #             print("No path found.")
        #         return

        # screen.fill((255, 255, 255))

        # # Draw the matrix on the screen
        # for i in range(matrix_size):
        #     for j in range(matrix_size):
        #         if matrix[i][j] == 1:
        #             pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif matrix[i][j] == 'R':
        #             pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif matrix[i][j] == 'P':
        #             pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
        #         elif (i, j) in closed_set:
        #             pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

        # pygame.display.flip()
        

        # Check if the goal has been reached
        if (x, y) == goal.position:
            path = []

            while current:
                path.append(current.position)
                # Check if the current node has a parent (not the start node)
                if current.parent:
                    # Custo base do movimento
                    cost = 1
                    # Se mudar de direção, adiciona um custo adicional
                    if current.parent.direction != current.direction:
                        cost = 2
                    total_cost += cost

                current = current.parent

            path.reverse()

            if path:
                print(f"\nPath: {path}")
                print(f"Total Cost: {total_cost}")
                path_found = True
                
                # while True:
                #     for event in pygame.event.get():
                #         if event.type == pygame.QUIT:
                #             pygame.quit()
                #             return

                #     screen.fill((255, 255, 255))

                #     for i in range(matrix_size):
                #         for j in range(matrix_size):
                #             if matrix[i][j] == 1:
                #                 pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                #             elif matrix[i][j] == 'R':
                #                 pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
                #             elif matrix[i][j] == 'P':
                #                 pygame.draw.rect(screen, (0, 0, 255), (j * cell_size, i * cell_size, cell_size, cell_size))
                #             elif (i, j) in path:
                #                 pygame.draw.rect(screen, (0, 255, 0), (j * cell_size, i * cell_size, cell_size, cell_size))
                #             elif (i, j) in closed_set:
                #                 pygame.draw.rect(screen, (255, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size))

                #     pygame.display.flip()
                #     #pygame.time.delay(10)

                #     for event in pygame.event.get():
                #         if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                #             pygame.quit()
                #             return path
                
            return path, total_cost

        # Check the neighbors of the current node
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if is_valid(new_x, new_y, matrix):
                if (new_x, new_y) not in closed_set:
                    neighbor = Node((new_x, new_y), parent=current, direction=(dx, dy))
                    neighbor.g = current.g + 1 
                    neighbor.h = heuristic(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h

                    if neighbor not in open_set:
                        open_set.append(neighbor)
                    else:
                        existing_neighbor = next((n for n in open_set if n == neighbor), None)
                        if existing_neighbor and neighbor.g < existing_neighbor.g:
                            existing_neighbor.g = neighbor.g
                            existing_neighbor.parent = current
                            existing_neighbor.f = existing_neighbor.g + existing_neighbor.h

    if not path_found:
        print("No path found.")
    #    pygame.quit()
    return None, None