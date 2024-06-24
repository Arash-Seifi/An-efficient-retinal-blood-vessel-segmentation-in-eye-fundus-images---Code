![image](https://github.com/Arash-Seifi/An-efficient-retinal-blood-vessel-segmentation-in-eye-fundus-images---Code/assets/92459590/f9a66873-818b-4446-b944-e7edf416e270)# Note:
This is not my article, and the code that I wrote is only my own understanding of the concepts explained in the paper.

# Background and objective:
Automatic segmentation of retinal blood vessels makes a major contribu￾tion in CADx of various ophthalmic and cardiovascular diseases. A procedure to segment thin and thick
retinal vessels is essential for medical analysis and diagnosis of related diseases. In this article, a novel
methodology for robust vessel segmentation is proposed, handling the existing challenges presented in
the literature.
# Methods: 
The proposed methodology consists of three stages, pre-processing, main processing, and post￾processing. The first stage consists of applying filters for image smoothing. The main processing stage
is divided into two configurations, the first to segment thick vessels through the new optimized top￾hat, homomorphic filtering, and median filter. Then, the second configuration is used to segment thin
vessels using the proposed optimized top-hat, homomorphic filtering, matched filter, and segmentation
using the MCET-HHO multilevel algorithm. Finally, morphological image operations are carried out in the
post-processing stage

![image](https://github.com/Arash-Seifi/An-efficient-retinal-blood-vessel-segmentation-in-eye-fundus-images---Code/assets/92459590/d6ee26ef-326d-487b-8344-850536b0afe1)

# My results:
The following image shows the image that I was able to achieve by following the instructions in the article. As it is shown, I couldn't replicate the exact code necessary for the proposed accuracy, but nevertheless, I could reach an acceptable image. 

# Instructions:
In the code directory, you will find two main files. The first "main.py" is used for calculating the actual output for the input RGB image. The only things you need to change are the input RGB image, the mask of the input, and the actual segmented result from a reputable repository such as DRIVE. "final.py" is used for calculating the accuracy and for the final image enhancement.
![image](https://github.com/Arash-Seifi/An-efficient-retinal-blood-vessel-segmentation-in-eye-fundus-images---Code/assets/92459590/b1bae7bc-9dfa-4d51-b8ae-b8857be5a285)
![image](https://github.com/Arash-Seifi/An-efficient-retinal-blood-vessel-segmentation-in-eye-fundus-images---Code/assets/92459590/b1bae7bc-9dfa-4d51-b8ae-b8857be5a285)
