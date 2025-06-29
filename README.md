# Image-Clustering
Image Clustering is an unsupervised machine learning technique that allows images to be clustered in groups based on similar visual properties and helps with a range of applications in content organization, retrieval, and anomaly detection. This work introduces a novel framework for image clustering using multimodal learning, leveraging the CLIP (Contrastive Language–Image Pretraining) model to combine image features with descriptive words. Traditional methods of image clustering based on unimodal algorithms often struggle with large datasets containing complex features. The proposed method bridges this gap by transforming unimodal image data into a multimodal framework through CLIP's descriptive capabilities. The use of text-based centroids that are optimized using k-means clustering algorithms has become the core of the method for enhancing cluster representation. These centroids introduce metadata-driven descriptions to the feature space, leading to a huge gain in performance. Experimental results confirm that clustering performance with these centroids approaches zero-shot distribution benchmarks, demonstrating the framework's robustness and adaptability across datasets such as CIFAR-10, STL10, and MIT Scene Recognition. While the method provides fast development and strong performance, scalability to large datasets is limited by clustering's computational complexity. This work promises much by vision-language models in unsupervised learning, with possibilities beyond traditional methods, and it opens the door to future innovation. Public resources supporting this methodology will encourage people to implement it even more. To address scalability and enhance feature quality, self-supervised pretraining methods such as SimCLR or BYOL can be integrated. These techniques offer robust feature representations, improving clustering accuracy and generalization, especially for complex datasets. Additionally, the LiT model can be used to enrich the image features further with semantic textual information, which provides richer and more context-aware representations. This multimodal approach improves the quality and interpretability of the learned features, especially in datasets involving both visual and text data.

# Process
The datasets used are CIFAR10 and STL10. Both the datasets used in the code are inbuilt and are imported.
Two different combinations of models are used to evaluate the datasets. They are CLIP + LiT and CLIP + SimCLR + BYOL.

# 1. CLIP + LiT
CLIP (Contrastive Learning - Image Pretraining)
CLIP is a vision-language model developed by OpenAI that can understand images in the context of natural language. It stands out because it was trained on a large dataset of image–text pairs from the internet, allowing it to learn powerful and generalizable representations of both images and text.

How CLIP works:
CLIP uses two encoders:
1.An image encoder (like a Vision Transformer or ResNet) that turns an image into a feature vector.
2.A text encoder (like a Transformer) that turns text into a feature vector.
These encoders are trained together using contrastive learning, so that the image and its matching caption get similar embeddings, while unrelated pairs are pushed apart.
At inference time, CLIP can compare any image with any text prompt by measuring cosine similarity between their embeddings.

What makes CLIP special:
1.Zero-shot capabilities: CLIP doesn’t need task-specific training—it can perform many tasks just by comparing text prompts to image embeddings.
2.Multimodal understanding: It learns to connect visual concepts with language, which allows it to classify, search, or describe images using only text.
3.Generalization: Because it's trained on diverse, web-scale data, CLIP often outperforms models trained on narrow, labeled datasets.

LiT (Locked Image Tuning)
