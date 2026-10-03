# Phase 3: Hybrid Ensemble Architecture & Justification

## 1. Model Selection: Why DenseNet121 and ResNet50?

For our final Hybrid Ensemble, we are selecting **DenseNet121** and **ResNet50** as our two base models. 

**Why these two?**
1. **DenseNet121 (The Texture Expert):** This model was our absolute best performer in Phase 2, achieving **68.22%** accuracy. DenseNet connects every layer to every other layer (Dense Connectivity), which makes it incredibly powerful at recognizing fine, local textures in the spectrograms—perfect for catching small crackles and wheezes.
2. **ResNet50 (The Global Expert):** This model was our strong runner-up, hitting **51.16%**. Its architecture uses Residual Skip Connections, allowing it to look at the "big picture" of the spectrogram without losing information deep in the network. It captures the overall structural timing of the breathing cycles.

**Why not the others?**
* **MobileNetV2** performed terribly (30.23%) because its lightweight depthwise-separable convolutions failed to capture the complexity of the augmented respiratory audio.
* **EfficientNet-B0** performed decently (46.51%), but it scales the image dimensions in a way that aggressively overfits on smaller medical datasets compared to the robust architecture of ResNet50.

## 2. Layer Selection: What are we fusing?

To push the accuracy past the **90% threshold**, we cannot simply average the final predictions. Instead, we are performing **Deep Feature Fusion** by extracting the feature maps directly from the deepest convolutional layers of both models.

* **From DenseNet121:** We extract from the very last block (`relu` activation after `conv5_block16`). This layer outputs a 1024-dimensional feature map containing the highest-level abstractions of the audio textures.
* **From ResNet50:** We extract from its final residual block (`conv5_block3_out`). This outputs a 2048-dimensional feature map representing the deep semantic meaning of the breathing cycle.

## 3. How the Hybrid Ensemble Works

1. **Parallel Processing:** When an audio spectrogram is inputted, it is fed into both DenseNet121 and ResNet50 simultaneously.
2. **Global Average Pooling (GAP):** We take the 2D feature maps from the selected layers and flatten them into 1D vectors using GAP. This gives us 1024 features from DenseNet and 2048 features from ResNet.
3. **Feature Concatenation:** We physically merge (concatenate) these vectors together into a massive **3072-dimensional super-vector**. This vector contains the combined "knowledge" of both models.
4. **Classification Head:** This super-vector is then passed through a new, fully connected neural network (Dense Layers). We will apply aggressive **Dropout (0.5)** to force the model to rely equally on both ResNet and DenseNet features.
5. **Final Output:** A 4-class Softmax layer makes the final diagnosis (Healthy, Pneumonia, URTI, Bronchiectasis).

## 4. Path to >90% Accuracy
By fusing the local texture recognition of DenseNet with the structural representation of ResNet, the network overcomes the weaknesses of individual models. Paired with our 1,354 augmented training samples from Phase 2, this hybrid architecture is specifically designed to maximize generalization on unseen patient data and break the 90% clinical accuracy barrier.
