---
layout: post
category: python machine_learning deep_learning 
---

Convolving an image with a filter makes the volume of the image shrink with each convolution layer. The dimensions of the resulting layer
is give by the formula

$$ n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1 $$


If we have for example a 32 $$x$$ 32 image. With a stride of 1 and a filter size of 3 $$x$$ 3 we will end up with a 30 $$x$$ 30 image.

Padding is a convenient way to maintain the image size and convolve an image without shrinking it. With padding the 
resulting image dimensions are give by 

$$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$

With a padding of 1 this convolution can have its dimensions preserved. 

General formula to determine padding is 

$$ p = \frac{ f - 1}{2} $$

If the height and width of the image is fully preserved then such a convolution is termed as **"same"** convolution.

Padding also helps in preserving the information from borders of the image. 

