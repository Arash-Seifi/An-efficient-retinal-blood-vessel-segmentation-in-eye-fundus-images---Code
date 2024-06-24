import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.metrics import mean_squared_error
from scipy.spatial.distance import dice
from sklearn.metrics import jaccard_score, confusion_matrix

segmented_image_path = 'Segmented_Test.jpg'
segmented_image = Image.open(segmented_image_path).convert('RGB')
mask_image_path = 'Image_Mask.gif'
mask_image = Image.open(mask_image_path).convert('L') # Convert to grayscale

segmented_data = np.array(segmented_image)
mask_data = np.array(mask_image)

# Invert the mask
mask_inverted = 255 - mask_data

# Dilate the inverted mask to create a thicker edge
dilated_mask = binary_dilation(mask_inverted, structure=np.ones((15, 15))).astype(mask_inverted.dtype) * 255

# Apply the dilated mask to the segmented image
masked_segmented_image = segmented_data.copy()
masked_segmented_image[dilated_mask == 255] = [0, 0, 0]  

# Save the final image
final_image_path = 'Final_Image.jpg'
final_image = Image.fromarray(masked_segmented_image)
final_image.save(final_image_path)
print(f"Final image saved as {final_image_path}")

# Load the images
image1 = Image.open("Final_Image.jpg").convert("L")
image2 = Image.open("33_manual1.gif").convert("L")

# Ensure images have the same size
if image1.size != image2.size:
    image2 = image2.resize(image1.size, Image.ANTIALIAS)

# Convert images to binary format
threshold = 128
image1 = image1.point(lambda p: p > threshold and 255)
image2 = image2.point(lambda p: p > threshold and 255)

# Convert images to NumPy arrays
arr1 = np.array(image1)
arr2 = np.array(image2)

# Convert 255 values to 1
arr1 = np.where(arr1 == 255, 1, 0)
arr2 = np.where(arr2 == 255, 1, 0)

# Flatten the arrays for easier calculation of metrics
flat_arr1 = arr1.flatten()
flat_arr2 = arr2.flatten()

# Calculate pixel-wise comparisons
tp = np.sum((flat_arr1 == 1) & (flat_arr2 == 1))
tn = np.sum((flat_arr1 == 0) & (flat_arr2 == 0))
fp = np.sum((flat_arr1 == 1) & (flat_arr2 == 0))
fn = np.sum((flat_arr1 == 0) & (flat_arr2 == 1))

# Total number of pixels
total_pixels = flat_arr1.size

# Specificity, Sensitivity, Accuracy
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
accuracy = (tp + tn) / total_pixels

# Calculate Mean Squared Error (MSE)
mse_value = mean_squared_error(flat_arr1, flat_arr2)
# Calculate Dice Coefficient
dice_coefficient = 1 - dice(flat_arr1, flat_arr2)
# Calculate Jaccard Index (Intersection over Union)
jaccard_index = jaccard_score(flat_arr1, flat_arr2)

# Display the results
print(f"MSE: {mse_value}")
print(f"Dice Coefficient: {dice_coefficient}")
print(f"Jaccard Index: {jaccard_index}")
print(f"Specificity: {specificity}")
print(f"Sensitivity: {sensitivity}")
print(f"Accuracy: {accuracy}")

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot known segmentation
axes[0].imshow(image2, cmap='gray')
axes[0].set_title('Known Segmentation')
axes[0].axis('off')

# Plot implemented segmentation
axes[1].imshow(image1, cmap='gray')
axes[1].set_title('Implemented Segmentation')
axes[1].axis('off')

plt.tight_layout()
plt.show()
