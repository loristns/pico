# 🤏Pico

Pico is a byte-level language model architecture that eliminates tokenization using
a Mixture-of-Depths routing mechanism.

![Pico architecture](assets/architecture.svg)

- Mixtures-of-Depths[^1] router as a dynamic multiscale modeling mechanism
- Multi-token prediction[^2] heads to unlock self-speculative decoding
at inference time, allowing the model to generate multiple tokens in parallel.
- Sliding window, grouped-query attention
- ALiBi positional encoding[^3]
- SwiGLU[^4]
- Training with SOAP optimizer[^5]

This architecture is inspired by [SpaceByte](https://github.com/kjslag/spacebyte)[^6],
but allows the model to learn when to insert latent blocks by itself in an end-to-end manner.

*Work in progress*.

[^1]: David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258) arXiv preprint arXiv:2404.02258, 2024.
[^2]: Fabian Gloeckle and Badr Youbi Idrissi and Baptiste Rozière and David Lopez-Paz and Gabriel Synnaeve. [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) arXiv preprint arXiv:2404.19737, 2024.
[^3]: Ofir Press and Noah A. Smith and Mike Lewis. [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) arXiv preprint arXiv:2108.12409, 2022.
[^4]: Noam Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) arXiv preprint arXiv:2002.05202, 2020.
[^5]: Nikhil Vyas and Depen Morwani and Rosie Zhao and Itai Shapira and David Brandfonbrener and Lucas Janson and Sham Kakade. [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321) arXiv preprint arXiv:2409.11321, 2024.
[^6]: Kevin Slagle. [SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/abs/2404.14408) arXiv preprint arXiv:2404.14408, 2024.