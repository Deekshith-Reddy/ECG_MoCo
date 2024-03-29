## Augmentations

[reference](https://arxiv.org/pdf/2007.15951.pdf)

### GaussianNoise(Jittering)

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 0 we create a noise in the gaussian distribution with standard deviation, mean and the noise is *added* to the ecg.


![Gaussian Noise](/images/gaussian_noise.png "Gaussian Noise")

This is the noise that we add to the original ECG to "jitter" it produces the signal below

![Gaussian Noise Comparison](/images/gaussianNoise_comparison.png "Gaussian Noise Comparison with original ECG")



### GaussianBlur

This augmentation is made by using scipy's gaussian_filter function where a gaussian kernel is created with a randomly picked standard deviation uniformly from the range(0.1, 0.25)


![Gaussian Blur Comparison](/images/gaussianBlur_comparison.png "Gaussian Blur Comparison with original ECG")


### Scaling

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 1 and we create magnitudes for each time instance in the gaussian distribution with standard deviation, mean and the magnitudes are *multiplied* to the ecg

![Scaling Magnitude](/images/scaling_mag.png "Scaling Magnitude")

The above is the noise for example that we use to multiply the ECG

![Scaling Comparison](/images/scaling_comparison.png "Scaling Comparison with original ECG")


### Magnitude Warping

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 1 we pick num_knots(10) of points to create a cubic spline and then we *multiply* it to the ecg, to create a augmented signal

![Magnitude Warping](/images/magnitude_warping.png "Magnitude Warping")

The above is the warping used to transform the ECG Signal below

![Magnitude Warping Comparison](/images/magnitudeWarping_comparison.png "Magnitude Warping Comparison with original ECG")

