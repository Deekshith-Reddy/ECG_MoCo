## Augmentations

[reference](https://arxiv.org/pdf/2007.15951.pdf)

#### GaussianNoise(Jittering)

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 0 we create a noise in the gaussian distribution with standard deviation, mean and *added* to the noise.


![Gaussian Noise](/images/gaussian_noise.png "Gaussian Noise")

This is the noise that we add to the original ECG to "jitter" it


