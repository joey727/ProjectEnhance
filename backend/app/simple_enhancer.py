import cv2
import numpy as np
from PIL import Image
import os


class SimpleImageEnhancer:
    def __init__(self):
        """Initialize the enhanced image enhancer with improved OpenCV-based methods."""
        pass

    def enhance_image(self, pil_image, enhancement_type="deblur"):
        """
        Enhance an image using improved OpenCV-based methods.

        Args:
            pil_image: PIL Image object
            enhancement_type: Type of enhancement ("deblur", "sharpen", "denoise", "enhance", "super_resolution")

        Returns:
            PIL Image: Enhanced image
        """
        # Convert PIL to OpenCV format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array (OpenCV format)
        img = np.array(pil_image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if enhancement_type == "deblur":
            enhanced = self._advanced_deblur_image(img)
        elif enhancement_type == "sharpen":
            enhanced = self._advanced_sharpen_image(img)
        elif enhancement_type == "denoise":
            enhanced = self._advanced_denoise_image(img)
        elif enhancement_type == "enhance":
            enhanced = self._comprehensive_enhance_image(img)
        elif enhancement_type == "super_resolution":
            enhanced = self._super_resolution_enhance(img)
        else:
            # Default to comprehensive enhancement
            enhanced = self._comprehensive_enhance_image(img)

        # Convert back to PIL format
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)

    def _advanced_deblur_image(self, img):
        """Apply conservative deblurring using gentle techniques."""
        # Convert to float
        img_float = img.astype(np.float32) / 255.0

        # Method 1: Gentle Wiener filter with smaller kernel
        kernel_size = 11  # Reduced for less aggressive deblurring
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size

        # Apply Wiener filter with higher noise estimation for gentler effect
        img_fft = np.fft.fft2(img_float, axes=(0, 1))
        kernel_fft = np.fft.fft2(kernel, s=img_float.shape[:2])

        # Higher noise power for gentler filtering
        noise_power = 0.02  # Increased for less aggressive filtering
        wiener_filter = np.conj(kernel_fft) / \
            (np.abs(kernel_fft)**2 + noise_power)
        wiener_filter = np.stack([wiener_filter] * 3, axis=2)

        # Apply filter
        result_fft = img_fft * wiener_filter
        result = np.real(np.fft.ifft2(result_fft, axes=(0, 1)))

        # Method 2: Gentle Richardson-Lucy deconvolution with fewer iterations
        psf = np.ones((9, 9)) / 81  # Smaller kernel
        result_rl = self._richardson_lucy_deconvolution(
            img_float, psf, iterations=5)  # Fewer iterations

        # Combine methods with more weight on original
        combined = 0.3 * result + 0.2 * result_rl + 0.5 * img_float

        # Clip and convert back to uint8
        combined = np.clip(combined, 0, 1) * 255
        return combined.astype(np.uint8)

    def _richardson_lucy_deconvolution(self, img, psf, iterations=10):
        """Implement Richardson-Lucy deconvolution algorithm."""
        # Initialize estimate
        estimate = img.copy()

        # Normalize PSF
        psf = psf / np.sum(psf)

        for _ in range(iterations):
            # Forward projection
            forward = cv2.filter2D(estimate, -1, psf)
            forward = np.clip(forward, 1e-6, 1.0)  # Avoid division by zero

            # Ratio
            ratio = img / forward

            # Backward projection
            backward = cv2.filter2D(ratio, -1, psf)

            # Update estimate
            estimate = estimate * backward

        return estimate

    def _advanced_sharpen_image(self, img):
        """Apply gentle sharpening using conservative techniques."""
        # Method 1: Gentle unsharp mask with reduced parameters
        gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)  # Reduced blur radius
        unsharp_mask = cv2.addWeighted(
            img, 1.3, gaussian, -0.3, 0)  # Reduced contrast

        # Method 2: Gentle Laplacian sharpening with smaller kernel
        kernel = np.array([[0, -0.5, 0],
                          [-0.5, 3, -0.5],
                          [0, -0.5, 0]])  # Reduced values
        laplacian_sharp = cv2.filter2D(img, -1, kernel)

        # Method 3: Bilateral filter for edge-preserving sharpening
        bilateral = cv2.bilateralFilter(img, 9, 50, 50)  # Reduced parameters
        bilateral_sharp = cv2.addWeighted(
            img, 1.2, bilateral, -0.2, 0)  # Reduced contrast

        # Combine methods with more weight on original image
        combined = cv2.addWeighted(unsharp_mask, 0.4, laplacian_sharp, 0.2, 0)
        combined = cv2.addWeighted(combined, 0.7, bilateral_sharp, 0.1, 0)
        # Add original image back
        combined = cv2.addWeighted(combined, 0.8, img, 0.2, 0)

        return np.clip(combined, 0, 255).astype(np.uint8)

    def _advanced_denoise_image(self, img):
        """Apply advanced denoising using multiple techniques."""
        # Method 1: Non-local Means with optimized parameters
        nlm_denoised = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)

        # Method 2: Bilateral filter for edge-preserving denoising
        bilateral_denoised = cv2.bilateralFilter(img, 15, 50, 50)

        # Method 3: Median filter for salt-and-pepper noise
        median_denoised = cv2.medianBlur(img, 3)

        # Method 4: Gaussian filter for smooth denoising
        gaussian_denoised = cv2.GaussianBlur(img, (3, 3), 0.8)

        # Combine methods adaptively
        # Use more aggressive denoising for dark areas, less for bright areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_mask = gray / 255.0

        # Weighted combination
        combined = (brightness_mask[:, :, np.newaxis] * nlm_denoised +
                    (1 - brightness_mask[:, :, np.newaxis]) * bilateral_denoised)

        # Add some median filtering for very noisy areas
        noise_mask = cv2.Laplacian(gray, cv2.CV_64F).var()
        if noise_mask > 100:  # High noise detected
            combined = cv2.addWeighted(combined, 0.7, median_denoised, 0.3, 0)

        return np.clip(combined, 0, 255).astype(np.uint8)

    def _comprehensive_enhance_image(self, img):
        """Apply balanced comprehensive image enhancement."""
        # Step 1: Gentle denoising
        denoised = self._advanced_denoise_image(img)

        # Step 2: Gentle contrast enhancement using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel with conservative parameters
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8))  # Reduced clip limit
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Step 3: Gentle saturation enhancement
        # Reduced saturation boost
        lab[:, :, 1] = np.clip(lab[:, :, 1] * 1.05, 0, 255)
        # Minimal color enhancement
        lab[:, :, 2] = np.clip(lab[:, :, 2] * 1.02, 0, 255)

        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Step 4: Apply gentle sharpening
        sharpened = self._advanced_sharpen_image(enhanced)

        # Step 5: Final gentle color correction
        # Convert to HSV for better color control
        hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)

        # Gentle brightness enhancement
        # Reduced brightness boost
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.02, 0, 255)

        # Gentle saturation enhancement
        # Reduced saturation boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)

        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend with original for more natural result
        final = cv2.addWeighted(final, 0.8, img, 0.2, 0)

        return final

    def _super_resolution_enhance(self, img):
        """Apply super-resolution-like enhancement."""
        # Upscale using bicubic interpolation
        height, width = img.shape[:2]
        upscaled = cv2.resize(img, (width * 2, height * 2),
                              interpolation=cv2.INTER_CUBIC)

        # Apply edge enhancement
        kernel = np.array([[-1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1],
                          [-1, -1, 25, -1, -1],
                          [-1, -1, -1, -1, -1],
                          [-1, -1, -1, -1, -1]])
        edge_enhanced = cv2.filter2D(upscaled, -1, kernel)

        # Apply bilateral filtering for noise reduction
        bilateral = cv2.bilateralFilter(edge_enhanced, 9, 75, 75)

        # Combine with original upscaled image
        enhanced = cv2.addWeighted(upscaled, 0.7, bilateral, 0.3, 0)

        # Downscale back to original size with high quality
        final = cv2.resize(enhanced, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)

        return final


class SimpleImagePredictor:
    """Wrapper class for the enhanced image enhancer."""

    def __init__(self):
        self.enhancer = SimpleImageEnhancer()

    def predict(self, pil_image, enhancement_type="enhance"):
        """
        Predict enhanced image using the enhanced enhancer.

        Args:
            pil_image: PIL Image object
            enhancement_type: Type of enhancement to apply

        Returns:
            PIL Image: Enhanced image
        """
        return self.enhancer.enhance_image(pil_image, enhancement_type)
