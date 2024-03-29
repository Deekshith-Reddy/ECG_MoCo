## Augmentations

[reference](https://arxiv.org/pdf/2007.15951.pdf)

### GaussianNoise(Jittering)

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 0 we create a noise in the gaussian distribution with standard deviation, mean and the noise is *added* to the ecg.


![Gaussian Noise](/images/GuassianNoise.png "Gaussian Noise")

This is the noise that we add to the original ECG to "jitter" it produces the signal below

![Gaussian Noise Comparison](/images/GuassianNoise_comparison.png "Gaussian Noise Comparison with original ECG")



### GaussianBlur

This augmentation is made by using scipy's gaussian_filter function where a gaussian kernel is created with a randomly picked standard deviation uniformly from the range(0.1, 0.25)


![Gaussian Blur Comparison](/images/GaussianBlur_comparison.png "Gaussian Blur Comparison with original ECG")


### Scaling

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 1 and we create magnitudes for each time instance in the gaussian distribution with standard deviation, mean and the magnitudes are *multiplied* to the ecg

![Scaling Magnitude](/images/Scaling.png "Scaling Magnitude")

The above is the noise for example that we use to multiply the ECG

![Scaling Comparison](/images/Scaling_comparison.png "Scaling Comparison with original ECG")


### Magnitude Warping

A random standard deviation between 0.1 and 0.25 is uniformly picked and with a mean 1 we pick num_knots(10) of points to create a cubic spline and then we *multiply* it to the ecg, to create a augmented signal

![Magnitude Warping](/images/MagnitudeWarping.png "Magnitude Warping")

The above is the warping used to transform the ECG Signal below

![Magnitude Warping Comparison](/images/MagnitudeWarping_comparison.png "Magnitude Warping Comparison with original ECG")


### Random Crop and Resize

A random number in the range (4000, 5000) is selected for the window of the signal to cropped, based on the window size the signal is either shrinked(window > 4500) or expanded(window < 4500) to get to 4500 time instances

In the below depiction 'None' values are padded in front and back of the transformed signal to demonstrate, otherwise in the original transformation, the transformed signal starts at 0 and ends at 4500.

![Random Crop Resize](/images/RandomCropResize_comparison.png "Random Crop Resize")

