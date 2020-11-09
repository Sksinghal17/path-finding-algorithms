import pygame as pg
from queue import PriorityQueue

SC_SIZE = 800
ROWS = 40

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)
PURPLE = (82, 0, 89)
L_GREY = (180, 180, 180)
GREY = (224, 224, 224)


WIN = pg.display.set_mode((SC_SIZE, SC_SIZE))
pg.display.set_caption("Dijkstra's Algorithm")


#--------------------------------------------------------------------------------------#


class Node:
    def __init__(self, r, c, width, total_rows):
        self.width = width
        # Position on the grid
        self.row = r
        self.col = c
        # Position in terms of pixels
        self.x = c * width
        self.y = r * width
        # Initial color of all cells is white
        self.color = WHITE
        self.sharing_border = []
        self.total_rows = total_rows

    def get_pos(self):
        # Get grid-wise position
        return (self.row, self.col)

    def is_closed(self):
        # A GREY cell is marked as closed.
        return self.color == GREY

    def is_open(self):
        # A Yellow cell is yet to be explored
        return self.color == YELLOW

    def is_barrier(self):
        # A BLACK cell is user defined barrier
        return self.color == BLACK

    def is_starting_pos(self):
        # A red cell is starting position
        return self.color == RED

    def is_ending_pos(self):
        # A BLUE cell is the final destination of path
        return self.color == BLUE

    def reset_color(self):
        self.color = WHITE

    def make_closed(self):
        # A GREY cell is marked as closed.
        if not self.is_starting_pos():
            self.color = GREY

    def make_open(self):
        # A Yellow cell is to be explored
        if self.is_ending_pos():
            return
        self.color = YELLOW

    def make_barrier(self):
        # A BLACK cell is user defined barrier
        self.color = BLACK

    def make_starting_pos(self):
        # A red cell is starting position
        self.color = RED

    def make_ending_pos(self):
        # A BLUE cell is the final destination of path
        self.color = BLUE

    def make_path(self):
        """ shortest path from node A --> B is shown in Green.
        color each node to green.
        """
        if self.is_ending_pos() or self.is_starting_pos():
            return
        self.color = GREEN

    def draw(self, WIN):
        """draw the node
        Args:
            WIN (pygame window)
        """
        if self.is_ending_pos() or self.is_starting_pos():
            radius = self.width // 2
            center = (self.x + radius, self.y + radius)
            pg.draw.circle(WIN, self.color, center, radius)
            return
        top_left_height_width = (self.x, self.y, self.width, self.width)
        pg.draw.rect(WIN, self.color, top_left_height_width)

    def update_surrounding(self, grid):
        """add neighbours to the surroundings list
        Args:
            grid (2d Node list)
        """
        # look down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.sharing_border.append(grid[self.row + 1][self.col])
        # look up
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.sharing_border.append(grid[self.row - 1][self.col])
        # look right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.sharing_border.append(grid[self.row][self.col + 1])
        # look left
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.sharing_border.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False
#--------------------------------------------------------------------------------------#


def H(p1, p2):
    """heuristic function
    Args:
        given any node n (with p1, p2 as coordinates of two nodes)
        h(n) is a heuristic function that estimates the cost of the cheapest path from n-node to the destination
    return: Manhattan distance of p1, p2
    """
    return 0
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = abs(x1 - x2), abs(y1 - y2)
    # a tie breaker is added to the result:
    # h(n) *= 1 + 1/p: p = (minimum cost of taking one step)/(expected maximum path length).
    return (dx + dy) * (1 + 1 / SC_SIZE)

#--------------------------------------------------------------------------------------#


def make_grid(rows, width):
    """making the grid data structure
    Args:
        rows (integer): how many rows we want?
        width (integer): screen width.
    """
    grid = []
    # Size of each Node
    gap = width // rows

    for r in range(rows):
        ls = []
        for c in range(rows):
            node = Node(r, c, gap, rows)
            ls.append(node)
        grid.append(ls)

    return grid

#--------------------------------------------------------------------------------------#


def draw_grid(win, rows, width):
    """draw grid lines
    Args:
        win (pygame window)
        rows (integer)
        width (integer)
    """
    gap = width // rows
    for r in range(rows):
        # Horizontal lines
        start = (0, r * gap)
        stop = (width, r * gap)
        pg.draw.line(win, L_GREY, start, stop)
        # Vertical lines
        start = (r * gap, 0)
        stop = (r * gap, width)
        pg.draw.line(win, L_GREY, start, stop)

#--------------------------------------------------------------------------------------#


def redraw(win, grid, rows, width):
    """draws the grid dynamically
    Args:
        win (pygame screen)
        grid (a 2d array of Nodes)
        rows (int)
        width (int)
    """
    # White paint everything
    win.fill(WHITE)
    # paint all nodes of the grid
    for row in grid:
        for node in row:
            node.draw(win)
    # paint grid lines
    draw_grid(win, rows, width)
    pg.display.update()

#--------------------------------------------------------------------------------------#


def get_clk_pos(pos, rows, width):
    """convert pixel pos to node pos
    Args:
        pos (tupple)
        rows (int)
        width (int)
    Returns:
        tupple
    """
    gap = width // rows
    x, y = pos
    px = x // gap
    py = y // gap
    return (py, px)

#--------------------------------------------------------------------------------------#


def empty_the_queue(pq):
    while not pq.empty():
        node = pq.get()[1]
        node.reset_color()

#--------------------------------------------------------------------------------------#


def make_final_path(came_from, current, draw):
    """Draw the final path
    Args:
        came_from (dict)
        current (Node): End node
        draw (function)
    """
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()

#--------------------------------------------------------------------------------------#


def find_path(draw, grid, start, end):
    """Dijkstra's Algorithm to find shortest path.
    It is same as A-star path finder, just without any Heuristic function
    we just count the steps to arrive at an current node and ignore the distance from current node -> end node
    Args:
        draw (function)
        grid (2d list)
        start (Node)
        end (Node)
    """
    pq = PriorityQueue()
    came_from = {}
    # given any node n,
    # g(n) = cost of start --> n
    # f(n) = g(n)
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = 0

    # initialize the queue
    # (f(n), n)
    pq.put((f_score[start], start))

    while not pq.empty():
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit()

        # poping the node with shortest f_score
        # 1 is the index of node in tupple we added to the queue
        current = pq.get()[1]
        if current == end:
            # we have found the path
            # no further consderation, so empty the queue
            empty_the_queue(pq)
            current.make_ending_pos()
            make_final_path(came_from, end, draw)
            return True

        for nbr in current.sharing_border:
            # cost of current --> nbr is 1, since we are neighbours
            tmp_g = g_score[current] + 1
            if tmp_g < g_score[nbr]:
                # we have found a better path to reach this neighbour from start node
                # we store the trace
                came_from[nbr] = current
                # update the g(n) of the neighbour
                g_score[nbr] = tmp_g
                # estimate h(n) of neighbout which is g(n) in Dijkstra
                f_score[nbr] = g_score[nbr]
                # add this data to the queue
                if True:  # nbr not in pq_items:
                    pq.put((f_score[nbr], nbr))
                    # we may explore this node in future
                    nbr.make_open()

        # We have considered this node
        current.make_closed()
        draw()

    return False

#--------------------------------------------------------------------------------------#


def main(win, width):
    grid = make_grid(ROWS, width)
    start_node = None
    end_node = None
    run = True

    while run:
        redraw(win, grid, ROWS, width)
        for e in pg.event.get():
            if e.type == pg.QUIT:
                run = False
                continue

            if pg.mouse.get_pressed()[0]:
                # Left click
                # get pixel pos
                pos = pg.mouse.get_pos()
                # get node pos
                row, col = get_clk_pos(pos, ROWS, width)
                # get clicked node
                node = grid[row][col]
                if not start_node and node != end_node:
                    start_node = node
                    start_node.make_starting_pos()
                elif not end_node and node != start_node:
                    end_node = node
                    end_node.make_ending_pos()
                elif node != end_node and node != start_node:
                    node.make_barrier()
            elif pg.mouse.get_pressed()[2]:
                # right click
                # reset the node state
                pos = pg.mouse.get_pos()
                # get node pos
                row, col = get_clk_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset_color()
                if node == end_node:
                    end_node = None
                elif node == start_node:
                    start_node = None

            if e.type == pg.KEYDOWN:
                # Enter space key to start searching the path
                if e.key == pg.K_SPACE and start_node and end_node:
                    # update the neighbors
                    for row in grid:
                        for node in row:
                            node.update_surrounding(grid)
                    # start searching for path
                    path_found = find_path(lambda: redraw(
                        win, grid, ROWS, width), grid, start_node, end_node)
                    if not path_found:
                        print("No Path Found.")
                if e.key == pg.K_c:
                    start_node, end_node = None, None
                    grid = make_grid(ROWS, width)
    pg.quit()


if __name__ == "__main__":
    main(WIN, SC_SIZE)