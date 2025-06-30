# Image-Clustering
Image Clustering is an unsupervised machine learning technique that allows images to be clustered in groups based on similar visual properties and helps with a range of applications in content organization, retrieval, and anomaly detection. This work introduces a novel framework for image clustering using multimodal learning, leveraging the CLIP (Contrastive Language–Image Pretraining) model to combine image features with descriptive words. Traditional methods of image clustering based on unimodal algorithms often struggle with large datasets containing complex features. The proposed method bridges this gap by transforming unimodal image data into a multimodal framework through CLIP's descriptive capabilities. The use of text-based centroids that are optimized using k-means clustering algorithms has become the core of the method for enhancing cluster representation. These centroids introduce metadata-driven descriptions to the feature space, leading to a huge gain in performance. Experimental results confirm that clustering performance with these centroids approaches zero-shot distribution benchmarks, demonstrating the framework's robustness and adaptability across datasets such as CIFAR-10, STL10, and MIT Scene Recognition. While the method provides fast development and strong performance, scalability to large datasets is limited by clustering's computational complexity. This work promises much by vision-language models in unsupervised learning, with possibilities beyond traditional methods, and it opens the door to future innovation. Public resources supporting this methodology will encourage people to implement it even more. To address scalability and enhance feature quality, self-supervised pretraining methods such as SimCLR or BYOL can be integrated. These techniques offer robust feature representations, improving clustering accuracy and generalization, especially for complex datasets. Additionally, the LiT model can be used to enrich the image features further with semantic textual information, which provides richer and more context-aware representations. This multimodal approach improves the quality and interpretability of the learned features, especially in datasets involving both visual and text data.


# Process
The datasets used are CIFAR10 and STL10. Both the datasets used in the code are inbuilt and are imported.

Two different combinations of models are used to evaluate the datasets. They are CLIP + LiT and CLIP + SimCLR + BYOL.


## CLIP (Contrastive Learning - Image Pretraining)
CLIP is a vision-language model developed by OpenAI that can understand images in the context of natural language. It stands out because it was trained on a large dataset of image–text pairs from the internet, allowing it to learn powerful and generalizable representations of both images and text.


> How CLIP works:

CLIP uses two encoders:

1.An image encoder (like a Vision Transformer or ResNet) that turns an image into a feature vector.

2.A text encoder (like a Transformer) that turns text into a feature vector.

These encoders are trained together using contrastive learning, so that the image and its matching caption get similar embeddings, while unrelated pairs are pushed apart.

At inference time, CLIP can compare any image with any text prompt by measuring cosine similarity between their embeddings.


> What makes CLIP special:

1.Zero-shot capabilities: CLIP doesn’t need task-specific training—it can perform many tasks just by comparing text prompts to image embeddings.

2.Multimodal understanding: It learns to connect visual concepts with language, which allows it to classify, search, or describe images using only text.

3.Generalization: Because it's trained on diverse, web-scale data, CLIP often outperforms models trained on narrow, labeled datasets.


## LiT (Locked Image Tuning)
LiT (Locked-image Tuning) is a fine-tuning method introduced by Google for improving vision-language models like CLIP, especially when transferring to new tasks with limited labeled data.

> What is LiT (Locked-image Tuning)?

LiT stands for Locked-image Tuning, a strategy where:

-> The image encoder is kept frozen (locked).

-> Only the text encoder is fine-tuned on a new dataset/task.

-> This is the opposite of traditional fine-tuning where you adjust all or most model weights. 

-> LiT was introduced to get the best of both worlds: the strong generalization ability of pretrained vision models and task-specific adaptation through text.


> How LiT works:

1.Start with a pretrained image encoder (e.g., from CLIP or another large vision model like ViT).

2.Keep the image encoder fixed—no weight updates.

3.Train or fine-tune the text encoder using a dataset of (image, label/text) pairs.

4.Use contrastive learning so that the image embedding matches the updated text embedding.


> Why use LiT?

1.More stable fine-tuning: Prevents overfitting or destroying learned visual features.

2.Efficient and fast: Training is faster since fewer parameters are updated.

3.Works well with limited data: Especially useful in low-data regimes.

4.Strong zero-shot performance: After LiT, models can still generalize well to unseen tasks using text prompts.


## 1.CLIP + LiT working

1. CLIP (Contrastive Language–Image Pretraining) uses two encoders—one for images and one for text—to map both into a shared embedding space. It learns to align images and their textual descriptions using contrastive learning.

2. LiT (Locked-image Tuning) enhances CLIP by freezing the pretrained image encoder and only fine-tuning the text encoder on a new dataset. This allows the model to adapt to new tasks while preserving the general visual knowledge already learned.

3. During training, text descriptions (prompts) are fine-tuned to better match the fixed image embeddings. This helps generate more semantic and meaningful representations for clustering or classification.

4. In clustering tasks, this combination allows for text-based centroids and more interpretable clusters, since images are grouped not just by pixels or features, but by semantic similarity via language.
