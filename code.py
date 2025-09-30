import requests

def render_unicode_grid(doc_url):
    # Fetch the data
    response = requests.get(doc_url)
    data = response.text.strip().splitlines()

    # Parse and store characters by coordinates
    grid = {}
    max_x = max_y = 0

    for line in data:
        if not line.strip():
            continue
        parts = line.strip().split(',')
        if len(parts) != 3:
            continue
        char, x, y = parts[0], int(parts[1]), int(parts[2])
        grid[(x, y)] = char
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Build and print the grid
    for y in range(max_y + 1):
        row = ""
        for x in range(max_x + 1):
            row += grid.get((x, y), ' ')
        print(row)