# Image inpainting using the latent space of StyleGAN

The goal of this project is to use StyleGAN to do image inpainting.

More precisely, we use StyleGAN2-Ada to make the inpaiting. The first idea consists in finding a latent code compatible with the parts of the image we can observe: $$w^* = \argmin\limits_w \| (x - G(w))\odot(1-M)\|^2_2$$
This problem is solved through gradient descent on the latent code $w$, with an initialization that is either random or given by an encoder.

We will look wether or not the model will produce something realistic inside the region, and wether or not it is robust to small variations.

Maybe we will build a gradio app to explore all of this, more interactively.

Once this first step is done, we will add more regularization options, to improve the quality of the inpainting. Possible regularizations can be based on LPIPS, (as proposed during the meeting), or adversarial (convenient, since we have access to StyleGAN2's critic).

We can also add semantic constraints: for example, we can constrain the inpainting of a mouth to be smiling, using a classifier (in this case, we would use the BCE of the class as a regularization) or a CLIP model.

More on CLIP: CLIP is made of two neural networks. The first is the textual encoder: it maps sentences to textual embeddings. The second is the image encoder, which maps the images to image embeddings. These two embeddings are in the same dimension, and match when their cosine similarity is close to 1. Therefore, one could add $-\text{cos}(E_I(G(w)), E_T(\text{phrase}))$ to the loss, to enforce the semantic constraint.



**Technical details**:

- The implementation of StyleGAN2 is directly downloaded from [this repository](https://github.com/NVlabs/stylegan2-ada-pytorch). Unfortunately, this code is copyrighted, and we invite the users to download it by themselves, as otherwise the authors of this repository would violate the copyrights.
