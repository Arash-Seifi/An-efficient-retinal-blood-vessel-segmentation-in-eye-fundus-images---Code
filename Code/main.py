import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.morphology import disk, white_tophat, remove_small_objects, binary_dilation, binary_erosion
from skimage.filters import median
from skimage import img_as_ubyte
from skimage.morphology import disk, opening, closing
from scipy.ndimage import rotate, convolve
from scipy.ndimage import median_filter

max_iter = 250
population_size = 30
num_thresholds = 4

# Initialize population of hawks (thresholds)
def initialize_hawks(population_size, num_thresholds, image):
    hawks = np.random.randint(0, 256, size=(population_size, num_thresholds))
    return hawks

# Cross-Entropy
def cross_entropy(thresholds, image):
    hist, _ = np.histogram(image, bins=np.arange(257))
    hist = hist / hist.sum()
    def calculate_Hk(start, end):
        prob = hist[start:end].sum()
        mean_intensity = np.dot(np.arange(start, end), hist[start:end]) / prob if prob > 0 else 0
        Hk = np.sum(hist[start:end] * np.log(mean_intensity + 1e-10))
        return Hk
    thresholds = np.sort(thresholds)
    thresholds = np.concatenate(([0], thresholds, [256]))
    H = 0
    for i in range(1, len(thresholds)):
        H += calculate_Hk(thresholds[i-1], thresholds[i])
    return H

# HHO algorithm
def hho(image, max_iter, population_size, num_thresholds):
    hawks = initialize_hawks(population_size, num_thresholds, image)
    best_hawk = hawks[0]
    best_fitness = cross_entropy(best_hawk, image)

    for iter in range(max_iter):
        for i in range(population_size):
            # Randomly generate a new candidate hawk (thresholds)
            new_hawk = np.clip(hawks[i] + np.random.randint(-10, 10, size=num_thresholds), 0, 255)
            new_fitness = cross_entropy(new_hawk, image)
            if new_fitness < best_fitness:
                best_hawk = new_hawk
                best_fitness = new_fitness

    return best_hawk


# Segment the image using the optimal thresholds
def segment_image(image, thresholds):
    thresholds = np.sort(thresholds)
    segmented_image = np.zeros_like(image)
    segmented_image[image <= thresholds[0]] = thresholds[0]
    for i in range(1, len(thresholds)):
        segmented_image[(image > thresholds[i-1]) & (image <= thresholds[i])] = thresholds[i]
    segmented_image[image > thresholds[-1]] = thresholds[-1]
    return segmented_image



def green_channel_extraction(image):
    return image[:, :, 1]


def top_hat_transformation(image,open,close):
    struct_element_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open, open))
    struct_element_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    image_complement = np.max(image) - image
    opened_complement = cv2.morphologyEx(image_complement, cv2.MORPH_OPEN, struct_element_open)
    closed_image = cv2.morphologyEx(opened_complement, cv2.MORPH_CLOSE, struct_element_close)
    # Optimized top-hat result
    optimized_top_hat_image = cv2.subtract(image_complement, closed_image)
    return optimized_top_hat_image

def matched_filter(image, sigma=0.8, kernel_size=(7, 7)):
    # Generate 1D Gaussian kernel
    kernel_x = cv2.getGaussianKernel(kernel_size[0], sigma)
    kernel_y = cv2.getGaussianKernel(kernel_size[1], sigma)
    kernel = np.outer(kernel_x, kernel_y)
    filtered_images = []
    # Define the range of angles for rotation
    angles = range(0, 183, 7)
    for angle in angles:
        # Rotate the kernel
        rotation_matrix = cv2.getRotationMatrix2D((kernel_size[0]//2, kernel_size[1]//2), angle, 1)
        rotated_kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size[0], kernel_size[1]))
        # Convolve the rotated kernel with the input image
        filtered_image = cv2.filter2D(image, -1, rotated_kernel)
        # Append the filtered image to the list
        filtered_images.append(filtered_image)
    # Take the maximum response over all orientations
    max_response = np.max(filtered_images, axis=0)
    return max_response


def homomorphic_filtering(image, sigma):
    image = np.float32(image)
    # Apply logarithmic transformation
    log_image = np.log1p(image)
    # Perform Fourier Transform
    dft = cv2.dft(log_image, flags=cv2.DFT_COMPLEX_OUTPUT)
    # Shift the zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # Define the Gaussian high-pass filter
    rows, cols = image.shape[:2]
    center_row, center_col = rows // 2, cols // 2
    x = np.linspace(-center_col, center_col - 1, cols)
    y = np.linspace(-center_row, center_row - 1, rows)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    h = 1 - np.exp(-(d**2) / (2 * sigma**2))
    # Apply the filter in the frequency domain
    filtered_shift = dft_shift * h[:,:,np.newaxis]
    # Perform Inverse Fourier Transform
    filtered = cv2.idft(np.fft.ifftshift(filtered_shift), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    # Undo the logarithmic transformation
    filtered_exp = np.expm1(filtered)
    # Convert back to uint8
    filtered_img = np.uint8(filtered_exp)
    return filtered_img

#def median(image, neighborhood):
    result = np.zeros_like(image)
    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood around the current pixel
            neighborhood_rows = slice(max(0, i - neighborhood[0]//2), min(rows, i + neighborhood[0]//2 + 1))
            neighborhood_cols = slice(max(0, j - neighborhood[1]//2), min(cols, j + neighborhood[1]//2 + 1))
            neighborhood_values = image[neighborhood_rows, neighborhood_cols]
            result[i, j] = np.median(neighborhood_values)

    return result

def post_process(binary_image, min_size=5):
    cleaned_image = remove_small_objects(binary_image.astype(bool), min_size=min_size)
    # Dilate
    cleaned_image = binary_dilation(cleaned_image, disk(1))
    # Erode
    cleaned_image = binary_erosion(cleaned_image, disk(1))
    return cleaned_image



def process_image(image_path):
    image = cv2.imread(image_path)
    #Part0: Preprocessing
    green_channel = green_channel_extraction(image)
    smoothed_image = gaussian_filter(green_channel, sigma=np.sqrt(0.463))
    
    #Part1: Thick
    thick_vessels = top_hat_transformation(smoothed_image, 12,12)
    thick_vessels = homomorphic_filtering(thick_vessels, 2)
    thick_vessels =  median_filter(thick_vessels, 1)
    #thick_vessels = top_hat_transformation(thick_vessels, 32,86)
    _, thick_vessels = cv2.threshold(thick_vessels, 1, 255, cv2.THRESH_BINARY)
    
    
    #Part2: Thin
    thin_vessels = top_hat_transformation(smoothed_image, 9, 20)
    thin_vessels = homomorphic_filtering(thin_vessels, 1)
    thin_vessels = gaussian_filter(thin_vessels, sigma=0.1)
    thin_vessels = matched_filter(thin_vessels)
    optimal_thresholds = hho(thin_vessels, max_iter, population_size, num_thresholds)
    thin_vessels = segment_image(thin_vessels, optimal_thresholds)
    _, thin_vessels = cv2.threshold(thin_vessels, 1, 255, cv2.THRESH_BINARY)


    combined_vessels = np.bitwise_or(thick_vessels > 0, thin_vessels > 0)
    final_segmentation = post_process(combined_vessels)
    return final_segmentation


if __name__ == "__main__":
    image_path = "Test5.tif"
    segmented_image = process_image(image_path)
    cv2.imwrite("Segmented_Test.jpg", img_as_ubyte(segmented_image))
