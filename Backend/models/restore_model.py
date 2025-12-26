import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try to import lama-cleaner, but provide fallback if not available
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config
    LAMA_AVAILABLE = True
except ImportError:
    LAMA_AVAILABLE = False
    print("Warning: lama-cleaner not available, using simple restoration")

# ======================================================
# FINAL RESTORER (LAMA AUTO MODEL)
# ======================================================
class FinalRestorer:
    def __init__(self):
        self.lama_available = LAMA_AVAILABLE
        if LAMA_AVAILABLE:
            try:
                # 🔥 LaMa model (auto-download handled internally)
                self.model = ModelManager(
                    name="lama",
                    device=DEVICE
                )
                self.config = Config(
                    ldm_steps=20,
                    hd_strategy="resize",
                    hd_strategy_resize_limit=1024,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LAMA model: {e}")
                print("Falling back to simple restoration")
                self.lama_available = False
        else:
            self.model = None
            self.config = None

    # --------------------------------------------------
    # SCRATCH DETECTION
    # --------------------------------------------------
    def detect_scratches(self, img):
        """Detect scratches and artifacts in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Use adaptive thresholding to detect scratches
        # This is more conservative and won't over-detect
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)
        
        # Only keep strong edges (scratches)
        _, mask = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 3)
        
        return mask
    
    # --------------------------------------------------
    # IMAGE ENHANCEMENT (ALTERNATIVE TO INPAINTING)
    # --------------------------------------------------
    def _enhance_image(self, img):
        """Enhance image quality using denoising and sharpening"""
        try:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Denoise the image
            denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel * 0.1)
            
            # Blend original and sharpened
            result = cv2.addWeighted(denoised, 0.7, sharpened, 0.3, 0)
            
            # Convert back to RGB
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return result
        except Exception as e:
            print(f"Enhancement failed: {e}")
            return img.copy()

    # --------------------------------------------------
    # MAIN RESTORE FUNCTION
    # --------------------------------------------------
    def restore(self, img_array):
        # Accept numpy array (RGB format from PIL)
        if isinstance(img_array, np.ndarray):
            img = img_array.copy()
            # Ensure it's uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # Ensure RGB format (PIL gives RGB, OpenCV expects BGR)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Already RGB from PIL
                pass
            elif len(img.shape) == 2:
                # Grayscale, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # If string path provided, load it
            img = cv2.imread(img_array)
            if img is None:
                raise ValueError(f"Could not load image from path: {img_array}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Validate image
        if img is None or img.size == 0:
            raise ValueError("Invalid image array provided")
        
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")

        # Resize safely
        MAX_SIZE = 1024
        scale = min(MAX_SIZE / h, MAX_SIZE / w, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 1️⃣ Scratch mask - only detect if image has visible scratches/degradation
        mask = self.detect_scratches(img_resized)
        
        # Limit mask to avoid over-inpainting (only inpaint if mask covers < 30% of image)
        mask_ratio = np.sum(mask > 0) / (new_h * new_w)
        
        # 2️⃣ RESTORATION STRATEGY
        if mask_ratio > 0.3:
            # Too many artifacts detected, use enhancement instead of inpainting
            print(f"Mask covers {mask_ratio*100:.1f}% of image, using enhancement instead")
            clean = self._enhance_image(img_resized)
        elif mask_ratio < 0.01:
            # Very few artifacts, just enhance the image
            clean = self._enhance_image(img_resized)
        else:
            # Moderate artifacts, try inpainting
            if self.lama_available and self.model is not None:
                try:
                    # Try different API methods for lama-cleaner
                    if hasattr(self.model, 'inpaint'):
                        clean = self.model.inpaint(img_resized, mask, self.config)
                    elif hasattr(self.model, '__call__'):
                        clean = self.model(img_resized, mask, self.config)
                    else:
                        # Try accessing the actual model inside ModelManager
                        if hasattr(self.model, 'model'):
                            clean = self.model.model(img_resized, mask, self.config)
                        else:
                            raise AttributeError("No valid inpaint method found")
                    
                    # Validate output
                    if clean is None or clean.size == 0:
                        raise ValueError("LAMA returned empty result")
                    if clean.shape[:2] != (new_h, new_w):
                        print(f"Warning: Output shape mismatch, resizing")
                        clean = cv2.resize(clean, (new_w, new_h))
                    
                    # Blend with original for natural look
                    clean = cv2.addWeighted(img_resized, 0.2, clean, 0.8, 0)
                        
                except Exception as e:
                    print(f"LAMA restoration failed: {e}, using simple restoration")
                    clean = self._simple_restore(img_resized, mask)
            else:
                # Use simple restoration method
                clean = self._simple_restore(img_resized, mask)
            
            # Final enhancement pass
            clean = self._enhance_image(clean)

        # Validate cleaned image
        if clean is None or clean.size == 0:
            print("Warning: Restoration returned empty, using original")
            clean = img_resized.copy()
        
        # Ensure we have a valid numpy array
        if not isinstance(clean, np.ndarray):
            print("Warning: Restoration returned non-array, using original")
            clean = img_resized.copy()

        # Ensure correct format
        if clean.dtype != np.uint8:
            # Handle float arrays (0-1 or 0-255 range)
            if clean.max() <= 1.0:
                clean = (clean * 255).astype(np.uint8)
            else:
                clean = np.clip(clean, 0, 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(clean.shape) == 2:
            clean = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        elif len(clean.shape) == 3:
            if clean.shape[2] == 1:
                clean = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
            elif clean.shape[2] != 3:
                print(f"Warning: Invalid channels {clean.shape[2]}, using original")
                clean = img_resized.copy()
        else:
            print(f"Warning: Invalid shape {clean.shape}, using original")
            clean = img_resized.copy()

        # Resize back to original size
        if clean.shape[:2] != (h, w):
            clean = cv2.resize(clean, (w, h))
        
        # Final validation - ensure shape is correct
        if len(clean.shape) != 3 or clean.shape[2] != 3:
            print(f"Warning: Final validation failed, shape: {clean.shape}")
            # Last resort: return original resized
            clean = cv2.resize(img_resized, (w, h))
            if len(clean.shape) == 2:
                clean = cv2.cvtColor(clean, cv2.COLOR_GRAY2RGB)
        
        # Final check
        if clean.shape != (h, w, 3) or clean.dtype != np.uint8:
            raise ValueError(f"Final image validation failed: shape={clean.shape}, dtype={clean.dtype}")
        
        return clean
    
    # --------------------------------------------------
    # SIMPLE RESTORATION (FALLBACK)
    # --------------------------------------------------
    def _simple_restore(self, img, mask):
        """Simple inpainting using OpenCV's inpainting algorithm"""
        try:
            # Validate inputs
            if img is None or img.size == 0:
                raise ValueError("Invalid input image")
            if mask is None or mask.size == 0:
                return img.copy()
            
            # Ensure image is uint8 RGB
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape: {img.shape}")
            
            # Convert mask to uint8 if needed
            if mask.dtype != np.uint8:
                mask = np.clip(mask, 0, 255).astype(np.uint8)
            
            # Ensure mask is 2D
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Ensure mask and image have same dimensions
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            # Use OpenCV's inpainting
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
            
            # Validate result
            if result is None or result.size == 0:
                raise ValueError("Inpainting returned empty result")
            
            # Convert back to RGB
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Ensure uint8
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"Simple restoration failed: {e}, returning original")
            return img.copy() if img is not None else np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------- Load Restoration Model ----------------
_restore_model_instance = None

def load_restore_model():
    global _restore_model_instance
    if _restore_model_instance is None:
        _restore_model_instance = FinalRestorer()
    return _restore_model_instance

