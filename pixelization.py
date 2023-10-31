from PIL import Image, ImageOps
import math
import numpy as np
import json

def main():
    # get license plate bounding box:
    with open('data/bounding_box.json', 'r') as bounding_box_data_file:
        bounding_box_data = json.load(bounding_box_data_file)

    # noise up each image 
    for carID, bounding_box, in bounding_box_data.items():

        # file name to open and file name to write out
        file_name = f'data/images/{carID}.png'

        # open image
        image = Image.open(file_name)

        # convert image to grayscale
        gray_image = ImageOps.grayscale(image)

        pixelate_one_image(carID, gray_image, bounding_box, "laplace")
        pixelate_one_image(carID, gray_image, bounding_box, "gaussian")


def pixelate_one_image(carID, gray_image, bounding_box, mode):
    # get bounding box dimensions
    xmin, xmax, ymin, ymax = bounding_box["xmin"], bounding_box["xmax"], bounding_box["ymin"], bounding_box["ymax"]
    license_plate_width = xmax - xmin
    license_plate_height = ymax - ymin
    
    # compute grid dimensions
    num_cells = 16
    epsilon = 0.5
    cell_width = math.ceil(license_plate_width/num_cells)
    cell_height = math.ceil(license_plate_height/num_cells)

    # Loop through each cell to change the color value of its pixels to the average value + noise
    for row in range(num_cells):
        for col in range(num_cells):
            cell_sum = 0

            # Loop through all pixels in each cell
            for x in range(cell_width):
                for y in range(cell_height):
                    pixel_x = xmin + col * cell_width + x
                    pixel_y = ymin + row * cell_height + y

                    # Bounds checking
                    if pixel_x < xmax and pixel_y < ymax:
                        cell_sum += gray_image.getpixel((pixel_x, pixel_y))

            # Calculate average color value in each cell
            num_pixels = min(cell_width, xmax - xmin) * min(cell_height, ymax - ymin)
            cell_average = cell_sum / num_pixels

            if mode == "laplace":
                # Add laplace noise
                lp_noise = np.random.laplace(0, 255 / (num_cells * num_cells * epsilon))
                cell_average += lp_noise
            if mode == "gaussian":
                # Add gaussian noise
                gaus_noise = np.random.normal(0, 255 / (num_cells * num_cells))
                cell_average += gaus_noise


            # Ensure the cell value is within [0, 255]
            cell_average = max(0, min(255, cell_average))

            # Assign the cell value to all pixels in the cell
            for x in range(cell_width):
                for y in range(cell_height):
                    pixel_x = xmin + col * cell_width + x
                    pixel_y = ymin + row * cell_height + y

                    # Bounds checking
                    if pixel_x < xmax and pixel_y < ymax:
                        gray_image.putpixel((pixel_x, pixel_y), round(cell_average))

    # Save pixelated laplace image
    if mode == "laplace":
        gray_image.save(f"output/laplace_noise_{carID}.png")
    # Save pixelated gaussian image
    if mode == "gaussian":
        gray_image.save(f"output/gaussian_noise_{carID}.png")


main()