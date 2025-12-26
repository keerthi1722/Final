# How the Model Detects Image Tampering

## Overview
The model uses a **Hybrid Quantum-Classical Neural Network** architecture that combines:
- **VGG16 CNN** (Classical Deep Learning) for feature extraction
- **Quantum Neural Network** for enhanced pattern recognition
- **Multi-scale Analysis** for pixel-level detection

---

## Model Architecture

### 1. **Feature Extraction (VGG16 CNN)**
```
Input Image (224x224x3) 
    ↓
VGG16 Feature Extractor (Pre-trained on ImageNet)
    ↓
Extracts 512 high-level features
    ↓
Adaptive Average Pooling
    ↓
512-dimensional feature vector
```

**What VGG16 looks for:**
- **Texture patterns** - Inconsistencies in texture that indicate tampering
- **Edge patterns** - Unnatural edges or boundaries
- **Color inconsistencies** - Mismatched color distributions
- **Compression artifacts** - JPEG compression differences in tampered regions
- **Lighting inconsistencies** - Shadows or lighting that don't match the scene

### 2. **Deep Neural Network (DNN)**
```
512 features → 128 neurons → 4 qubits
```
- Reduces features to 4 dimensions for quantum processing
- Learns complex relationships between features

### 3. **Quantum Neural Network (QNN)**
```
4 qubits → Quantum Circuit → 4 quantum measurements
```
- Uses **quantum entanglement** to detect subtle patterns
- Can find correlations that classical networks might miss
- Particularly good at detecting:
  - **Subtle overlays** (like shadow overlays)
  - **Pixel-level inconsistencies**
  - **Complex tampering patterns**

### 4. **Final Classification**
```
Quantum output + DNN output → Final layer → 2 outputs
    ↓
[Authentic probability, Tampered probability]
```

---

## Detection Process

### Step 1: Image Preprocessing
1. **Resize** to 224x224 pixels
2. **Normalize** pixel values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
3. **Convert** to tensor format

### Step 2: Multi-Scale Analysis
The model analyzes the image at **3 different scales**:
- **100%** (original) - Weight: 50%
- **90%** (slightly smaller) - Weight: 25%
- **110%** (slightly larger) - Weight: 25%

**Why multi-scale?**
- Catches tampering at different resolutions
- Detects pixel-level inconsistencies
- More robust to image variations

### Step 3: Feature Extraction
VGG16 extracts features that capture:
- **Spatial patterns** - How pixels relate to neighbors
- **Frequency patterns** - Compression artifacts, noise patterns
- **Semantic features** - Object boundaries, textures

### Step 4: Quantum Processing
The quantum layer processes features to detect:
- **Non-local correlations** - Patterns that span across the image
- **Subtle inconsistencies** - Small differences that indicate tampering
- **Complex relationships** - Multi-dimensional patterns

### Step 5: Probability Calculation
The model outputs two probabilities:
- **P(Authentic)** - Probability image is real
- **P(Tampered)** - Probability image is tampered

These probabilities sum to 1.0 (100%)

### Step 6: Decision Making
The model uses **weighted averaging** across scales, then applies thresholds:

**For "Tampered" classification:**
- Requires: `P(Tampered) > 55%` AND `P(Tampered) - P(Authentic) > 8%`
- OR: `P(Tampered) > 50%` AND `P(Tampered) - P(Authentic) > 12%`

**For "Authentic" classification:**
- If `P(Authentic) > P(Tampered)` AND `P(Authentic) > 50%`
- Otherwise defaults to "Authentic" to reduce false positives

---

## What Tampering Signs the Model Detects

### 1. **Copy-Move Tampering**
- Duplicated regions (like duplicate deer shadows)
- Inconsistent textures in copied areas
- Edge artifacts from copy-paste operations

### 2. **Splicing/Compositing**
- Unnatural boundaries between regions
- Lighting inconsistencies
- Color mismatches between composited parts

### 3. **Shadow/Overlay Tampering**
- **Fake shadows/overlays** - Duplicate shadows, shadows that don't match lighting
- **Natural shadows** - Shadows that belong to objects (gradual, soft edges) are **AUTHENTIC**
- The model distinguishes:
  - ✅ **Natural shadows** (authentic): Gradual transitions, soft edges, consistent with lighting
  - ❌ **Fake shadows** (tampered): Sharp edges, abrupt transitions, duplicate shadows, inconsistent lighting

### 4. **Pixel-Level Manipulation**
- Unusual pixel value distributions
- Compression artifacts in specific regions
- Noise pattern inconsistencies

### 5. **Geometric Inconsistencies**
- Perspective mismatches
- Scale inconsistencies
- Distortion patterns

---

## Visual Highlighting (For Tampered Images)

When an image is detected as "Tampered", the system highlights suspicious regions using:

1. **Edge Detection** (Canny) - Detects sharp transitions
2. **Laplacian Operator** - Finds inconsistencies in image structure
3. **Brightness Variations** - Detects shadows and overlays
4. **Adaptive Thresholding** - Catches subtle shadow patterns
5. **Color Channel Analysis** - Finds RGB inconsistencies

These are combined to create a **red overlay** showing potential tampered regions.

---

## Model Training

The model (`hybrid_vgg16_quantum_ela.pt`) was trained on:
- **Authentic images** - Real, unmodified photographs
- **Tampered images** - Images with various types of tampering:
  - Copy-move
  - Splicing
  - Shadow overlays
  - Object removal/addition
  - Color manipulation

The model learned to distinguish between these patterns during training.

---

## Types of Images the Model Can Classify

### ✅ **What the Model Detects as "Tampered":**

1. **Manually Tampered Images:**
   - Copy-move tampering (duplicated objects/regions)
   - Image splicing/compositing (combining parts from different images)
   - Shadow/overlay manipulation (adding fake shadows)
   - Object removal or addition
   - Color manipulation in specific regions

2. **AI-Generated Images:**
   - **May be detected as "Tampered"** if they show:
     - Inconsistent textures (common in AI-generated images)
     - Unnatural patterns or artifacts
     - Compression inconsistencies
     - Lighting/shadow inconsistencies
     - Pixel-level anomalies typical of AI generation

3. **Deepfake Images:**
   - Face swaps or manipulations
   - Inconsistent facial features
   - Lighting mismatches

4. **Photoshopped Images:**
   - Any manual editing that creates inconsistencies
   - Retouching that leaves artifacts

### ✅ **What the Model Detects as "Authentic":**

1. **Real Photographs:**
   - Unmodified camera-captured images
   - **Natural shadows that belong to objects** (one object = one shadow)
   - Natural lighting and shadows with gradual transitions
   - Consistent textures and patterns
   - Natural compression artifacts

2. **Images with Natural Shadows:**
   - If an image has **one object with its natural shadow** → **Authentic**
   - Natural shadows have:
     - Gradual transitions (soft edges)
     - Consistent with lighting direction
     - Proper shadow softness
     - No duplicate or inconsistent shadows

2. **High-Quality AI-Generated Images:**
   - **May be detected as "Authentic"** if:
     - Very high quality (minimal artifacts)
     - Consistent patterns throughout
     - No obvious tampering signs
     - Well-generated lighting and textures

### ⚠️ **Important Notes:**

1. **AI-Generated Images:**
   - The model was **NOT specifically trained** to detect AI-generated images
   - It detects **tampering/manipulation patterns**, not AI generation per se
   - AI images with artifacts/inconsistencies → Likely "Tampered"
   - High-quality AI images → May be "Authentic"

2. **Model Focus:**
   - Primary purpose: Detect **manual tampering/manipulation**
   - Secondary: May catch AI-generated images with visible artifacts
   - **Not a dedicated AI image detector**

3. **For AI Image Detection:**
   - Would need a model specifically trained on:
     - Real photos vs. AI-generated images
     - Different AI models (DALL-E, Midjourney, Stable Diffusion, etc.)
     - Various generation techniques

---

## Limitations

1. **Model Quality**: The model's accuracy depends on:
   - Quality of training data
   - Types of tampering it was trained on
   - Image quality and resolution

2. **Subtle Tampering**: Very sophisticated tampering might not be detected if:
   - It matches the training data patterns too closely
   - It's done with professional tools
   - The tampering is extremely subtle

3. **False Positives/Negatives**: 
   - Some authentic images might be flagged (false positive)
   - Some tampered images might be missed (false negative)
   - This is why confidence scores are provided

---

## Improving Detection

To improve detection accuracy:
1. **Retrain the model** with more diverse tampering examples
2. **Fine-tune thresholds** based on your specific use case
3. **Add more training data** with examples similar to your images
4. **Use ensemble methods** - Combine multiple models

---

## Summary

The model detects tampering by:
1. ✅ Extracting visual features using VGG16
2. ✅ Processing features through quantum-enhanced neural network
3. ✅ Analyzing images at multiple scales
4. ✅ Looking for inconsistencies in:
   - Textures
   - Edges
   - Colors
   - Compression patterns
   - Lighting/shadow patterns
5. ✅ Making a binary decision: "Tampered" or "Authentic"

The confidence score indicates how certain the model is about its prediction.

