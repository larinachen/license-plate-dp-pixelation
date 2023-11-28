from PIL import Image, ImageOps
import math
import numpy as np
import json
import matplotlib.pyplot as plt

def main(num_cells, epsilon):
    # get license plate bounding box:
    with open('data/bounding_box.json', 'r') as bounding_box_data_file:
        bounding_box_data = json.load(bounding_box_data_file)

    # store mses
    none_mses = []
    laplace_mses = []
    gaussian_mses = []

    # noise up each image 
    for carID, bounding_box, in bounding_box_data.items():

        # file name to open and file name to write out
        file_name = f'data/images/{carID}.png'

        # open image
        image = Image.open(file_name)

        # convert image to grayscale
        gray_image = ImageOps.grayscale(image)
        gray_image.save(f"output/gray/gray_{carID}.png")

        # pixelation
        pixelate_one_image(carID, gray_image, bounding_box, "laplace", num_cells, epsilon)
        pixelate_one_image(carID, gray_image, bounding_box, "gaussian", num_cells, epsilon)
        pixelate_one_image(carID, gray_image, bounding_box, "none", num_cells, epsilon)

        # Calculate and print MSE values
        original_image = gray_image.crop((bounding_box["xmin"], bounding_box["ymin"], bounding_box["xmax"],  bounding_box["ymax"]))
        none_image = Image.open(f"output/pixelated/no_noise_pizelated{carID}.png")
        laplace_image = Image.open(f"output/laplace/laplace_noise_{carID}.png")
        gaussian_image = Image.open(f"output/gaussian/gaussian_noise_{carID}.png")
        mse_none = calculate_mse(original_image, none_image)
        mse_laplace = calculate_mse(original_image, laplace_image)
        mse_gaussian = calculate_mse(original_image, gaussian_image)

        none_mses.append(mse_none)
        laplace_mses.append(mse_laplace)
        gaussian_mses.append(mse_gaussian)

    none_avg = sum(none_mses)/len(none_mses)
    laplace_avg = sum(laplace_mses)/len(laplace_mses)
    gaussian_avg = sum(gaussian_mses)/len(gaussian_mses)

    print(f"finished processing for num_cells = {num_cells}, epsilon = {epsilon}")

    return none_avg, laplace_avg, gaussian_avg

def pixelate_one_image(carID, gray_image, bounding_box, mode, num_cells, epsilon):
    # get bounding box dimensions
    xmin, xmax, ymin, ymax = bounding_box["xmin"], bounding_box["xmax"], bounding_box["ymin"], bounding_box["ymax"]
    license_plate_width = xmax - xmin
    license_plate_height = ymax - ymin
    
    # compute grid dimensions
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
            elif mode == "gaussian":
                # Add gaussian noise
                gaus_noise = np.random.normal(0, 255 / (num_cells * num_cells))
                cell_average += gaus_noise
            elif mode == "none":
                # if neither, return just pixelated image
                pass

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
        gray_image.save(f"output/laplace/laplace_noise_{carID}.png")
        # crop out license plate images
        license_plate_image = gray_image.crop((xmin - 10, ymin -10, xmax + 10, ymax + 10))
        license_plate_image.save(f"output/cropped/cropped_laplace/laplce_licence_place{carID}.png")
    # Save pixelated gaussian image
    elif mode == "gaussian":
        gray_image.save(f"output/gaussian/gaussian_noise_{carID}.png")
        # crop out license plate images
        license_plate_image = gray_image.crop((xmin - 10, ymin - 10, xmax + 10, ymax + 10))
        license_plate_image.save(f"output/cropped/cropped_gaussian/gaussian_licence_place{carID}.png")
    elif mode == "none":
        gray_image.save(f"output/pixelated/no_noise_pizelated{carID}.png")
        # crop out license plate images
        license_plate_image = gray_image.crop((xmin - 10, ymin - 10, xmax + 10, ymax + 10))
        license_plate_image.save(f"output/cropped/cropped_pixelated/no_noise_pixelated{carID}.png")


def calculate_mse(original_image, pixelated_image):

    mse_value = 0

    for x in range(original_image.width):
        for y in range(original_image.height):
            mse_value += (original_image.getpixel((x, y)) - pixelated_image.getpixel((x, y))) ** 2

    mse_value /= (original_image.width * original_image.height)
    return mse_value

def plot_mse(laplace_avgs, gaussian_avgs, none_avgs, m_s, filename, title, x_axis, y_axis):
    # Create a line plot for Laplace MSE
    plt.plot(m_s, laplace_avgs, label='Laplace', marker='o', linestyle='-')

    # Create a line plot for Gaussian MSE
    plt.plot(m_s, gaussian_avgs, label='Gaussian', marker='x', linestyle='-')

    # Create a line plot for no noise MSE
    plt.plot(m_s, none_avgs, label='No noise', marker='^', linestyle='-')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()

    plt.savefig(filename)
    plt.show()

def plot_mse_diff(laplace_avgs, gaussian_avgs, none_avgs, m_s, filename, title, x_axis, y_axis):
    # diffs = np.array(laplace_avgs) - np.array(gaussian_avgs)
    lap_diffs = np.array(laplace_avgs) - np.array(none_avgs)
    gaus_diffs = np.array(gaussian_avgs) - np.array(none_avgs)
    plt.plot(m_s, lap_diffs, label='Laplace - None', marker='o', linestyle='-')
    plt.plot(m_s, gaus_diffs, label='Gaussian - None', marker='x', linestyle='-')
    plt.axhline(0, color='red', linestyle='--', label='y=0')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.legend()

    plt.savefig(filename)
    plt.show()

def m_experiments():
    laplace_avgs = []
    gaussian_avgs = []
    none_avgs = []
    m_s = [4, 8, 16, 32, 64, 128]

    for m in m_s:
        none_avg, laplace_avg, gaussian_avg = main(m, 0.5)

        none_avgs.append(none_avg)
        laplace_avgs.append(laplace_avg)
        gaussian_avgs.append(gaussian_avg)

    plot_mse(laplace_avgs, gaussian_avgs, none_avgs, m_s, 'output/plots/mse_against_m.png', 'Laplace and Gaussian MSE vs Number of Cells', 'Number of Cells', 'MSE')
    plot_mse_diff(laplace_avgs, gaussian_avgs, none_avgs, m_s, 'output/plots/diff_mse_against_m.png', 'MSE Difference vs Number of Cells', 'Number of Cells', 'MSE Difference')


def e_experiments():
    laplace_avgs = []
    gaussian_avgs = []
    none_avgs = []
    epsilons = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    for e in epsilons:
        none_avg, laplace_avg, gaussian_avg = main(16, e)

        none_avgs.append(none_avg)
        laplace_avgs.append(laplace_avg)
        gaussian_avgs.append(gaussian_avg)

    plot_mse(laplace_avgs, gaussian_avgs, none_avgs, epsilons, 'output/plots/mse_against_e.png', 'Laplace and Gaussian MSE vs Epsilon', 'Epsilon', 'MSE')
    plot_mse_diff(laplace_avgs, gaussian_avgs, none_avgs, epsilons, 'output/plots/diff_mse_against_e.png', 'MSE Difference vs Epsilon', 'Epsilon', 'MSE Difference')

# m_experiments()
e_experiments()