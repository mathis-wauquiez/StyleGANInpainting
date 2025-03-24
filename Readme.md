# Image inpainting using the latent space of StyleGAN

The goal of this project is to use StyleGAN to do image inpainting.

More precisely, we use StyleGAN2-Ada to make the inpaiting. The first idea consists in finding a latent code compatible with the parts of the image we can observe: 
$$w^* = \text{arg}\min\limits_w \| (x - G(w))\odot(1-M)\|^2_2$$

This problem is solved through gradient descent on the latent code $w$, with an initialization that is either random or given by an encoder. Alternatives are to use a different optimization algorithm, such as $\texttt{Adam}$ or $\texttt{L-BFGS}$, or to use different similarity loss (more about that later).

This code investigates wether or not the model produces something realistic inside the region, coherent with the information outside the area to inpaint, and wether or not the inpainting is robust to small noises in the $W^+$ space and to small modifications to the mask. 

This code also investigates more regularization options, to improve the quality of the inpainting. More precisely, we investagate LPIPS and an adversarial loss.

We can also add semantic constraints: for example, we can constrain the inpainting of a mouth to be smiling, using a classifier (in this case, we would use the BCE of the class as a regularization) or a CLIP model.

More on CLIP: CLIP is made of two neural networks. The first is the textual encoder: it maps sentences to textual embeddings. The second is the image encoder, which maps the images to image embeddings. These two embeddings are in the same dimension, and match when their cosine similarity is close to 1. Therefore, one could add $-\text{cos}(E_I(G(w)), E_T(\text{phrase}))$ to the loss, to enforce the semantic constraint.



**Technical details**:

- The implementation of StyleGAN2 is directly downloaded from [this repository](https://github.com/NVlabs/stylegan2-ada-pytorch). Unfortunately, this code is copyrighted, and we invite the users to download it by themselves, as otherwise the authors of this repository would violate the copyrights.

- I cannot share the code or weights for the classifier I used.