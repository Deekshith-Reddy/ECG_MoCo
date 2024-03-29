## Augmentations

[reference](https://arxiv.org/pdf/2007.15951.pdf)

#### GaussianNoise(Jittering)

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 0 we create a noise in the gaussian distribution with standard deviation, mean and the noise is *added* to the ecg.


![Gaussian Noise](/images/gaussian_noise.png "Gaussian Noise")

This is the noise that we add to the original ECG to "jitter" it produces the signal below

![Gaussian Noise Comparison](/images/gaussianNoise_comparison.png "Gaussian Noise Comparison with original ECG")



#### GaussianBlur

This augmentation is made by using scipy's gaussian_filter function where a gaussian kernel is created with a randomly picked standard deviation uniformly from the range(0.1, 0.25)


![Gaussian Blur Comparison](/images/gaussianBlur_comparison.png "Gaussian Blur Comparison with original ECG")


