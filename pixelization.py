from PIL import Image, ImageOps
import math
import numpy as np

# file name to open and file name to write out
file_name = 'murphy.png'
save_as = 'gray_pixelated.png'

# open image
image = Image.open(file_name)

# convert image to grayscale
gray_image = ImageOps.grayscale(image)

# save grayscale image for reference
gray_image.save('grey_murphy.png')


# get picture dimensions
width = gray_image.size[0]
height = gray_image.size[1]

# compute grid dimensions
num_cells = 15
cell_width = math.trunc(width/num_cells)
cell_height = math.trunc(height/num_cells)

cells = [[0 for r in range(num_cells)] for c in range(num_cells)]

cur_width = 0
cur_height = 0

# loop through all cells
for row in range(num_cells):
    for col in range(num_cells):

        # loop through all pixels in each cell
        for x in range(cell_width):
            for y in range(cell_height):
                cells[row][col] += gray_image.getpixel((cur_width + x, cur_height + y))

        # calculate average color value in each cell
        cells[row][col] = cells[row][col] / (cell_width * cell_height)

        m = cell_width * cell_height
        noise = np.random.laplace(0, 255 / (num_cells*num_cells))
        print(noise)
        cells[row][col] += noise

        if(cells[row][col] > 255):
            cells[row][col] = 255

        cur_width += cell_width

    cur_height += cell_height
    cur_width = 0


cur_width = 0
cur_height = 0

# loop through all cells
for row in range(num_cells):
    for col in range(num_cells):

        # loop through all pixels in each cell
        for x in range(cell_width):
            for y in range(cell_height):

                # set each pixel to the average color of this cell
                gray_image.putpixel((cur_width + x, cur_height + y), round(cells[row][col]))

        cur_width += cell_width
       
    cur_height += cell_height
    cur_width = 0

# save pixelated image
gray_image.save("noise_image.png")

